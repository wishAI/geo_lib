"""One bounded whole-body training run followed by serial, independent M2/M3 checks.

No simulator runs concurrently and no certificate is granted by this adapter.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def commands(args, output):
    checkpoint=output/'training'/args.name/('model_initial.pt' if getattr(args,'evaluate_initial',False) else f'model_{args.iterations-1}.pt')
    train=['--name',args.name,'--checkpoint',str(args.parent),'--whole-body',
           '--nominal-pose-profile','upright_waist_v1','--num-envs','256','--iterations',str(args.iterations),
           '--lift-noise','.1','--action-noise','.05']
    if args.residual_parent:train+=['--residual-checkpoint',str(args.residual_parent)]
    if getattr(args,'landing_quality',False):train+=['--landing-quality']
    if getattr(args,'training_seconds',None) is not None:train+=['--seconds',str(args.training_seconds)]
    train+=['--target-speed',str(getattr(args,'training_speed',.01))]
    if getattr(args,'curriculum_start_checkpoint',None):
        train+=['--curriculum-start-checkpoint',str(args.curriculum_start_checkpoint),
                '--curriculum-start-speed',str(args.curriculum_start_speed)]
    stages=[('train','leg_residual_training',train)]
    if getattr(args,'with_passive',False):
        stages.append(('passive','landau_rsl_control',['--name',args.name+'_passive','--mode','passive',
            '--profile','anatomical_legs_mass_v2','--nominal-pose-profile','upright_waist_v1','--render']))
    walking=['--name',args.name+'_walk','--checkpoint',str(checkpoint)]
    walking+=['--speed-target',str(args.speed_target)] if getattr(args,'speed_target',None) is not None else ['--seconds','90']
    return checkpoint,stages+[
        ('stand','leg_forward_control',['--name',args.name+'_stand','--checkpoint',str(checkpoint),'--stand']),
        ('walk','leg_forward_control',walking)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--parent',required=True,type=Path)
    p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--residual-parent',type=Path)
    p.add_argument('--training-speed',type=float,default=.01)
    p.add_argument('--training-seconds',type=float,help='Explicit episode horizon to include later support-transfer failures')
    p.add_argument('--landing-quality',action='store_true',help='Use the bounded touchdown-quality multiplier during training only')
    p.add_argument('--speed-target',type=float,choices=(.1,.4,.8,1.2),help='Use a fixed40s independent speed evaluation after standing')
    p.add_argument('--curriculum-start-checkpoint',type=Path)
    p.add_argument('--curriculum-start-speed',type=float,default=.006)
    p.add_argument('--evaluate-initial',action='store_true',help='Independently evaluate the exact frozen-screen actor with the new gait, before subsequent PPO updates')
    p.add_argument('--with-passive',action='store_true',help='Record a fresh passive-standing prerequisite after the checkpoint is created')
    args=p.parse_args()
    if args.evaluate_initial and not args.residual_parent:raise ValueError('Initial-policy evaluation requires a learned residual parent')
    if not .002<=args.training_speed<=1.2:raise ValueError('Training speed outside walking curriculum')
    if args.training_seconds is not None and not 10<=args.training_seconds<=60:raise ValueError('Training horizon outside bounds')
    if not re.fullmatch(r'anatomical_[A-Za-z0-9_]{1,80}',args.name) or not 2<=args.iterations<=200:
        raise ValueError('Campaign exceeds bounds')
    from . import mujoco_backend as backend
    args.parent=args.parent.resolve();args.parent.relative_to((backend.OUTPUT/'training').resolve())
    folder=backend.OUTPUT/(args.name+'_campaign');folder.mkdir(exist_ok=False)
    checkpoint,stages=commands(args,backend.OUTPUT)
    record={'status':'running','active_stage':None,'checkpoint':str(checkpoint),'evaluations':[],
            'parent_checkpoint':str(args.parent),'parent_checkpoint_sha256':backend.digest(args.parent),
            'canonical_milestone_pass':False,'source_sha256':backend.digest(__file__)}
    record['evaluated_weights']='exact_residual_parent' if args.evaluate_initial else 'updated_ppo'
    (folder/'campaign_source.py').write_text(Path(__file__).read_text())
    try:
        for stage,module,extra in stages:
            record['active_stage']=stage;backend.write_json(folder/'campaign.json',record)
            subprocess.run([sys.executable,'-m','algorithms.urdf_learn_wasd_walk.'+module,*extra],
                           check=True,timeout=3600 if stage=='train' else 300)
            if stage=='train':continue
            directory=backend.OUTPUT/(args.name+'_'+stage);result=json.loads((directory/'dynamics.json').read_text())
            record['evaluations'].append({'stage':stage,'directory':str(directory),'failures':result['failures'],
                                           'metrics':result['metrics']})
            if result['failures']:
                record.update(status='numerical_failure',active_stage=None)
                backend.write_json(folder/'campaign.json',record);return
        record.update(status='numerically_passed_visual_review_pending',active_stage=None)
    except BaseException as error:
        record.update(status='failed',error=str(error));raise
    finally:
        backend.write_json(folder/'campaign.json',record)

if __name__=='__main__':main()
