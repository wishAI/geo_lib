"""Full-duration foot closeups and dense, timestamped frame review packs."""
import argparse
import json
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from landau import Robot
from quality import measure
from retarget import load_source
from render import FONT,encoder_command,arrow
from state import OUT,sha256,write_json


def foot_video(run):
    r=Robot();src,sk=load_source(run/'source.npz');ix={s:i for i,(s,_) in enumerate(sk)}
    ret=json.loads((run/'retarget.json').read_text());C=np.asarray(ret['source_coordinate_matrix'])
    source=src['posed_joints']@C.T*ret['root_trajectory_scale']
    with np.load(run/'target.npz') as d:q=d['q'];base=d['base'];times=d['times']
    n=len(q);fps=round(1/np.median(np.diff(times)));font=ImageFont.truetype(FONT,15)
    proc=subprocess.Popen(encoder_command(run/'feet_proof.mp4',960,540,fps),stdin=subprocess.PIPE)
    feet=['foot_l','toes_01_l','foot_r','toes_01_r']
    srcfeet=['LeftFoot','LeftToeBase','RightFoot','RightToeBase']
    # Source ground is already in world coordinates. Use ankle mean XY only.
    try:
        for f in range(n):
            im=Image.new('RGB',(960,540),'#eef2f4');draw=ImageDraw.Draw(im)
            tf=r.fk(q[f],base[f]);meshes=[(l,v,faces) for l,v,faces in r.vertices(tf) if l in feet]
            for row in (0,1):
                pos=source[f,[ix[x] for x in srcfeet]] if row==0 else np.array([tf[x][:3,3] for x in feet])
                center=pos[[0,2]].mean(0)
                for col,axis in enumerate([0,1]):
                    ox=col*480;oy=row*250+32
                    def project(v):
                        v=np.asarray(v);return np.stack([ox+240+(v[...,axis]-center[axis])*800,oy+200-v[...,2]*800],axis=-1)
                    draw.rectangle((ox+2,oy,ox+478,oy+242),outline='#b6c4ce')
                    draw.line([(ox+3,oy+200),(ox+477,oy+200)],fill='#478861',width=2)
                    draw.text((ox+8,oy+5),('SOMA scaled' if row==0 else 'Landau soles')+(' | XZ from -Y' if col==0 else ' | YZ from +X'),font=font,fill='#344956')
                    if row:
                        polygons=[]
                        for link,v,faces in meshes:
                            xy=project(v);z=v[faces,1 if axis==0 else 0].mean(1)*(-1 if axis==0 else 1)
                            for face,depth in zip(faces,z):polygons.append((depth,xy[face],link))
                        for _,xy,link in sorted(polygons,key=lambda x:x[0]):
                            draw.polygon([tuple(v) for v in xy],fill='#6599b9' if link.endswith('_l') else '#b781aa',outline='#526878')
                    for k,side in [(0,'L'),(2,'R')]:
                        color='#126893' if k==0 else '#a04286'
                        a,b=project(pos[k]),project(pos[k+1]);arrow(draw,a,b,color,side+' toe',font)
                        if not row:
                            end=('Left' if k==0 else 'Right')+'ToeEnd'
                            if end in ix:draw.line([tuple(b),tuple(project(source[f,ix[end]]))],fill=color,width=3)
            draw.text((12,7),f'Foot close-up | frame {f+1}/{n} | {times[f]:.3f}s | same fixed axes, ankle-following translation',font=font,fill='#273e4d')
            proc.stdin.write(im.tobytes())
    finally:proc.stdin.close()
    if proc.wait():raise RuntimeError('Foot video encoder failed')


def decode(path):
    p=subprocess.Popen(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-of','json','-show_entries','stream=width,height,nb_frames',str(path)]))['streams'][0]
    width,height=int(probe['width']),int(probe['height']);size=width*height*3;frames=[]
    try:
        while True:
            raw=p.stdout.read(size)
            if not raw:break
            if len(raw)!=size:raise RuntimeError('Truncated decoded frame')
            frames.append(Image.frombytes('RGB',(width,height),raw))
    finally:p.stdout.close()
    if p.wait() or len(frames)!=int(probe['nb_frames']):raise RuntimeError('Incomplete video decode')
    return frames


def sheets(frames,selected,folder,prefix,cell=(384,288)):
    font=ImageFont.truetype(FONT,16);paths=[]
    for page,start in enumerate(range(0,len(selected),24)):
        indices=selected[start:start+24];sheet=Image.new('RGB',(cell[0]*4,(cell[1]+24)*6),'#eff3f5')
        draw=ImageDraw.Draw(sheet)
        for i,f in enumerate(indices):
            x=(i%4)*cell[0];y=(i//4)*(cell[1]+24)
            sheet.paste(frames[f].resize(cell),(x,y));draw.text((x+6,y+cell[1]+3),f'frame {f} | {f/30:.3f}s',font=font,fill='#233c4b')
        path=folder/f'{prefix}_{page:02}.png';sheet.save(path);paths.append({'path':str(path),'frames':indices})
    return paths


def prepare(run):
    report=measure(run);foot_video(run)
    clean=decode(run/'clean_preview.mp4');feet=decode(run/'feet_proof.mp4')
    folder=run/'review';folder.mkdir(exist_ok=True)
    equal=report['evenly_spaced_review_frames'];extras=sorted(set(report['review_frames'])-set(equal))
    pages=sheets(clean,equal,folder,'evenly24')+sheets(clean,extras,folder,'worst_transitions')
    footpages=sheets(feet,report['review_frames'],folder,'feet',cell=(384,216))
    # Every decoded frame is available for sequential, full-duration review.
    sequence=sheets(clean,list(range(len(clean))),folder,'full_sequence')
    value={'frame_count':len(clean),'full_duration_decode':'passed','playback_speed':'original 30fps',
        'clean_video_sha256':sha256(run/'clean_preview.mp4'),'foot_video_sha256':sha256(run/'feet_proof.mp4'),
        'selected_frames':report['review_frames'],'evenly_spaced_frames':equal,'selected_pages':pages,
        'foot_pages':footpages,'full_sequence_pages':sequence,'visual_review_status':'pending',
        'review_method_note':'Dense frame and full-sequence inspection can be recorded separately from real-time player playback; do not conflate them.'}
    write_json(run/'review.json',value)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');args=p.parse_args()
    for name in args.runs:
        result=prepare(OUT/'runs'/name);print(name,len(result['selected_frames']),'selected frames; all frames decoded')
