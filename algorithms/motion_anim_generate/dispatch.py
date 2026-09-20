"""Submit one bounded job to the parent-provisioned worker; never changes the service."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
ENV={**os.environ,'XDG_RUNTIME_DIR':'/run/user/1000','DBUS_SESSION_BUS_ADDRESS':'unix:path=/run/user/1000/bus'}


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['probe','unconditional-smoke','prompt-smoke'])
    p.add_argument('--action',choices=['idle','walk','turn','wave'],default='idle');p.add_argument('--embeddings',type=Path)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--seconds',type=float,default=2)
    p.add_argument('--steps',type=int,default=30);args=p.parse_args()
    active=subprocess.check_output(['systemctl','--user','list-units','motion-anim-gpu@*.service','--state=activating,active','--no-legend','--plain'],env=ENV,text=True)
    if active.strip():raise RuntimeError('Motion worker already active; inspect results before another job: '+active)
    job='motion'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    run=f'{args.command.replace("-","_")}_{args.action}_seed{args.seed}_{job}'
    worker_args=['--probe']
    if args.command!='probe':
        worker_args=['--run-id',run,'--seed',str(args.seed),'--seconds',str(args.seconds),'--steps',str(args.steps)]
        if args.command=='unconditional-smoke':worker_args+=['--unconditional']
        else:
            if not args.embeddings:raise ValueError('--embeddings is required for exact-prompt inference')
            worker_args+=['--embeddings',str(args.embeddings.resolve()),'--action',args.action]
    jobs=OUT/'backend_gpu/jobs';jobs.mkdir(parents=True,exist_ok=True)
    request={'kind':'module','module':'algorithms.motion_anim_generate.gpu_worker','args':worker_args,
             'timeout_s':120 if args.command=='probe' else 1800}
    with (jobs/(job+'.json')).open('x') as f:json.dump(request,f,indent=2)
    subprocess.run(['systemctl','--user','start',f'motion-anim-gpu@{job}.service'],env=ENV,check=True)
    print(json.dumps({'job':job,'run_id':run,'request':request,'result':str(OUT/'backend_gpu/results'/f'{job}.json')}))

if __name__=='__main__':main()
