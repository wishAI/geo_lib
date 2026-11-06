"""Publish bounded worker progress to the GUI outside the physics process."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from .evolution import write_evolution, ALGORITHM_ROOT


def motion_label(metrics):
    duration=float(metrics.get('duration_s',0))
    if 'semantic_forward_displacement_m' in metrics:
        motion=f"{float(metrics['semantic_forward_displacement_m']):.3f} m forward"
    else:motion=f"{1000*float(metrics.get('horizontal_drift_m',0)):.1f} mm standing drift"
    if metrics.get('sustained_forward_speed_mps') is not None:
        motion+=f" · sustained {float(metrics['sustained_forward_speed_mps']):.4f} m/s"
    return f"Recorded {duration:g} s · {motion} · falls {metrics.get('fall_count',0)} · resets {metrics.get('reset_count',0)}; certification separate."


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True);p.add_argument('--interval',type=float,default=30.)
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.job) or not 10<=args.interval<=60:raise ValueError('Invalid monitor bounds')
    worker=ALGORITHM_ROOT/'outputs/tk2-backend-20260918/source/algorithms/urdf_learn_wasd_walk/outputs/backend_gpu'
    request=json.loads((worker/'jobs'/(args.job+'.json')).read_text())
    arguments=request.get('args',[])
    name=arguments[arguments.index('--name')+1] if '--name' in arguments else None
    deadline=time.monotonic()+request['timeout_s']+120
    while time.monotonic()<deadline:
        path=worker/'results'/(args.job+'.json')
        try:result=json.loads(path.read_text())
        except (OSError,ValueError):result={}
        if result.get('state')=='running' and name:
            campaign_path=worker.parent/'mujoco'/(name+'_campaign')/'campaign.json'
            stage=None
            if campaign_path.exists():
                stage=json.loads(campaign_path.read_text()).get('active_stage')
            directory=worker.parent/'mujoco'/(name+'_'+stage if stage in ('passive','stand','walk') else name)
            if stage=='train':directory=worker.parent/'mujoco/training'/name
            evidence=next((directory/filename for filename in ('dynamics.json','training.json','metadata.json','progress.json')
                           if (directory/filename).exists()),directory/'progress.json')
            if evidence.exists():
                ledger_path=ALGORITHM_ROOT/'milestones.json';ledger=json.loads(ledger_path.read_text())
                session=ledger.get('sessionStatus',{});experiment=session.get('currentExperiment',{})
                active=experiment.get('activeTraining',{})
                if active.get('name')==name:
                    active['stage']=stage or 'evaluation'
                    experiment['evidence']=str(evidence.relative_to(ALGORITHM_ROOT.parents[1]))
                    experiment['status']='Policy training running' if stage=='train' else 'Independent '+str(stage or 'walking')+' evaluation running'
                    session['state']='training' if stage=='train' else 'evaluating';session['updatedAt']=datetime.now(timezone.utc).isoformat()
                    temporary=ledger_path.with_suffix('.monitor.tmp');temporary.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n');temporary.replace(ledger_path)
        payload=write_evolution()
        print(json.dumps({'job':args.job,'state':result.get('state','pending'),'current':payload['currentNodeId'],
                          'generated_at':datetime.now(timezone.utc).isoformat()}),flush=True)
        if result.get('state') in ('completed','failed'):
            ledger_path=ALGORITHM_ROOT/'milestones.json'
            ledger=json.loads(ledger_path.read_text());session=ledger.get('sessionStatus',{})
            experiment=session.get('currentExperiment',{});active=experiment.get('activeTraining',{})
            if active.get('name')==name:
                state='completed' if result.get('exit_code')==0 else 'failed'
                active.update(state=state,status=state)
                session['state']='evaluation_pending' if state=='completed' else 'diagnosis'
                session['updatedAt']=datetime.now(timezone.utc).isoformat()
                experiment['nextStep']='Independent no-reset evaluation and cumulative validation required.'
                campaign_path=worker.parent/'mujoco'/(str(name)+'_campaign')/'campaign.json'
                if campaign_path.exists():
                    campaign=json.loads(campaign_path.read_text())
                    evaluations=campaign.get('evaluations',[])
                    if evaluations:
                        last=evaluations[-1];directory=Path(last['directory'])
                        experiment['evidence']=str((directory/'dynamics.json').relative_to(ALGORITHM_ROOT.parents[1]))
                        experiment['motion']=str((directory/'motion.json').relative_to(ALGORITHM_ROOT.parents[1]))
                        experiment['video']=str((directory/'proof.mp4').relative_to(ALGORITHM_ROOT.parents[1]))
                        experiment['motionLabel']=experiment['videoLabel']=motion_label(last['metrics'])
                        experiment['lastDiagnostic']={'name':directory.name,'state':campaign['status'],'reason':'; '.join(last['failures']) or 'Numerical checks passed; cumulative certification and full visual review pending.'}
                        session['state']='diagnosis' if last['failures'] else 'review_pending'
                else:
                    directory=worker.parent/'mujoco'/str(name)
                    screen_path=worker.parent/'mujoco/training'/str(name)/'training.json'
                    if screen_path.exists():
                        screen=json.loads(screen_path.read_text())
                        experiment['evidence']=str(screen_path.relative_to(ALGORITHM_ROOT.parents[1]))
                        if screen.get('policy_family')=='anatomical_periodic_feedback_cem' and screen.get('metrics',{}).get('eligible') is False:
                            experiment['lastDiagnostic']={'name':str(name),'state':'screen_ineligible',
                                'reason':'No candidate met every replicated screening constraint; exported parameters remain diagnostic, with no milestone pass.'}
                            session['state']='diagnosis'
                    dynamics=directory/'dynamics.json'
                    if dynamics.exists():
                        evaluation=json.loads(dynamics.read_text());failures=evaluation.get('failures',[])
                        experiment['evidence']=str(dynamics.relative_to(ALGORITHM_ROOT.parents[1]))
                        for media in ('motion.json','proof.mp4'):
                            if (directory/media).exists():experiment['motion' if media=='motion.json' else 'video']=str((directory/media).relative_to(ALGORITHM_ROOT.parents[1]))
                        experiment['motionLabel']=experiment['videoLabel']=motion_label(evaluation.get('metrics',{}))
                        experiment['lastDiagnostic']={'name':str(name),'state':'numerical_failure' if failures else 'review_pending',
                            'reason':'; '.join(failures) or 'Numerical checks passed; cumulative certification and visual review pending.'}
                        session['state']='diagnosis' if failures else 'review_pending'
                if session['state']=='diagnosis':
                    experiment['status']='Run finished; walking gate remains unresolved'
                    experiment['summary']=experiment.get('lastDiagnostic',{}).get('reason','Worker failed; inspect the recorded result.')
                    experiment['nextStep']='Diagnose the recorded failure and continue the earliest unresolved milestone.'
                elif session['state']=='review_pending':
                    experiment['status']='Evaluation finished; independent review pending'
                    experiment['summary']='Numerical checks passed. Full visual review and cumulative certification are still required.'
                else:
                    experiment['status']='Training finished; evaluation pending'
                session['nextWork']=experiment['nextStep']
                temporary=ledger_path.with_suffix('.monitor.tmp');temporary.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n');temporary.replace(ledger_path)
                write_evolution()
            return
        time.sleep(args.interval)
    raise TimeoutError('Worker monitor exceeded the declared job deadline')

if __name__=='__main__':main()
