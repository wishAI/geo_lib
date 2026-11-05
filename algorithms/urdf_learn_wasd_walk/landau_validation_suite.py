"""Serial nominal validation jobs; stop at the first numerical failure."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    from algorithms.urdf_learn_wasd_walk import mujoco_backend as backend
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--turn-repeats',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.turn_repeats<=5:parser.error('Use 1..5 prospectively declared turn repeats')
    checkpoint=Path(args.checkpoint).resolve()
    checkpoint.relative_to((backend.OUTPUT/'training').resolve())
    folder=backend.OUTPUT/args.name
    folder.mkdir(exist_ok=False)
    record=dict(checkpoint=str(checkpoint),checkpoint_sha256=backend.digest(checkpoint),
                status='running',canonical_milestone_pass=False,evaluations=[])
    stages=[('stand',['--forward','0','--seconds','30']),
            ('forward5',['--forward','.2','--seconds','45']),
            ('forward10',['--forward','.2','--seconds','80','--target-distance','10'])]
    for index in range(args.turn_repeats):
        stages.append(('turn' if index==0 else 'turn_repeat'+str(index+1),
                       ['--forward','.2','--seconds','56','--turn']))
    stages.append(('teleop',['--forward','.2','--seconds','60','--teleop']))
    for direction in ('forward','left','right','backward'):
        stages.append((direction,['--forward','.2','--seconds','240','--target-distance','10',
                                  '--direction',direction,'--direction-navigation','hysteretic_v6']))
    for stage,extra in stages:
        record['active_stage']=stage
        backend.write_json(folder/'suite.json',record)
        name=args.name+'_'+stage
        subprocess.run([sys.executable,'-m','algorithms.urdf_learn_wasd_walk.landau_forward_control',
                        '--mode','evaluate','--name',name,'--checkpoint',str(checkpoint),*extra],check=True)
        result=json.loads((backend.OUTPUT/name/'dynamics.json').read_text())
        record['evaluations'].append(dict(stage=stage,directory=str(backend.OUTPUT/name),
                                         status=result['status'],failures=result['failures']))
        if result['failures']:
            record.update(status='numerical_failure',active_stage=None)
            backend.write_json(folder/'suite.json',record)
            print(json.dumps(record),flush=True)
            return
    record.update(status='numerically_passed_visual_review_pending',active_stage=None)
    backend.write_json(folder/'suite.json',record)


if __name__=='__main__':main()
