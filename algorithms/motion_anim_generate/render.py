"""CPU orthographic front/side evidence renderer of actual source joints and URDF meshes."""
import json
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from landau import Robot
from retarget import C, load_source
from state import sha256, write_json

W,H=960,960
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def render(run, label):
    r=Robot();src,skeleton=load_source(run/'source.npz')
    source=src['posed_joints']@C.T
    with np.load(run/'target.npz') as f:data={k:f[k] for k in f.files}
    report=json.loads((run/'validation.json').read_text())
    q=data['q'];base=data['base'];times=data['times'];n=len(times);fps=round(1/np.median(np.diff(times)))
    byframe={i:[] for i in range(n)}
    for v in report['violations']:
        if v['frame']>=0:byframe[v['frame']].append(v['check'])
    font=ImageFont.truetype(FONT,16);small=ImageFont.truetype(FONT,13)
    all_bounds=[]
    for f in [0,n//2,n-1]:
        all_bounds.extend([v for _,v,_ in r.vertices(r.fk(q[f],base[f]))])
    target_height=max(1.2,np.max(np.concatenate(all_bounds)[:,2])+.08)
    source_height=max(1.85,float(source[:,:,2].max())+.12)
    picked=set(np.linspace(0,n-1,6,dtype=int).tolist());shots=[]
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','24','-pix_fmt','yuv420p',str(run/'proof.mp4')]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    name_index={a:i for i,(a,_) in enumerate(skeleton)}
    edges=[(name_index[parent],i) for i,(_,parent) in enumerate(skeleton) if parent is not None]
    try:
        for f in range(n):
            canvas=Image.new('RGB',(W,H),'#101923');draw=ImageDraw.Draw(canvas)
            tf=r.fk(q[f],base[f]);meshes=r.vertices(tf)
            for row,(kind,height) in enumerate([('SOMA source',source_height),('Landau target',target_height)]):
                root=source[f,0] if row==0 else tf['root_x'][:3,3]
                for col,axis in enumerate([0,1]):
                    ox=col*480;oy=row*430+82;scale=350/height
                    def project(points):
                        a=np.asarray(points)
                        return np.stack([ox+240+(a[...,axis]-root[axis])*scale,oy+360-a[...,2]*scale],axis=-1)
                    draw.rectangle([ox+4,oy,ox+476,oy+408],fill='#e9eff2')
                    for z in np.arange(0,height,.1 if row else .25):
                        yy=oy+360-z*scale;draw.line([ox+6,yy,ox+474,yy],fill='#d1dbe0')
                    draw.line([ox+6,oy+360,ox+474,oy+360],fill='#528367',width=2)
                    draw.text((ox+12,oy+8),f'{kind} | {"front (X/Z)" if col==0 else "side (Y/Z)"}',font=font,fill='#152530')
                    if row==0:
                        xy=project(source[f])
                        for a,b in edges:draw.line([tuple(xy[a]),tuple(xy[b])],fill='#366889',width=4)
                        for pt in xy:draw.ellipse([pt[0]-2,pt[1]-2,pt[0]+2,pt[1]+2],fill='#17394f')
                        for k,foot in enumerate(['LeftFoot','RightFoot']):
                            pxy=xy[name_index[foot]]
                            contacts=src.get('foot_contacts')
                            contact=contacts is not None and bool(np.any(contacts[f,k*2:k*2+2]))
                            draw.ellipse([pxy[0]-5,pxy[1]-5,pxy[0]+5,pxy[1]+5],outline='#10964d' if contact else '#db8e22',width=2)
                    else:
                        polygons=[]
                        for link,vertices,faces in meshes:
                            projected=project(vertices);depth_axis=1 if axis==0 else 0
                            depths=vertices[faces,depth_axis].mean(1)
                            for face,depth in zip(faces,depths):
                                color=(83,127,149) if link.endswith('_l') else (153,170,181) if link.endswith('_r') else (120,139,150)
                                polygons.append((depth,projected[face],color))
                        for _,points,color in sorted(polygons,key=lambda v:v[0]):draw.polygon([tuple(v) for v in points],fill=color,outline=(70,89,102))
                        for side in ('l','r'):
                            pos=tf['foot_'+side][:3,3];pxy=project(pos)
                            bottom=min(v[:,2].min() for l,v,_ in meshes if l in ('foot_'+side,'toes_01_'+side))
                            draw.ellipse([pxy[0]-6,pxy[1]-6,pxy[0]+6,pxy[1]+6],outline='#e13136' if bottom<-.005 else '#10964d' if bottom<=.012 else '#db8e22',width=3)
                    draw.text((ox+12,oy+382),f'root XYZ: {root[0]:+.2f}, {root[1]:+.2f}, {root[2]:+.2f} m',font=small,fill='#243a49')
            draw.text((12,7),label[:108],font=font,fill='white')
            draw.text((12,29),f'frame {f+1}/{n} | {times[f]:.3f}s / {n/fps:.3f}s | original timing {fps}fps | kinematic only',font=font,fill='#b5cbd7')
            warning=', '.join(sorted(set(byframe[f]))) or 'no frame-local flags'
            draw.text((12,51),warning[:116],font=small,fill='#ff9388' if byframe[f] else '#90d2a4')
            draw.text((12,945),'Green: contact heuristic | orange: airborne | red: penetration | axes in metres; camera follows root',font=small,fill='white')
            encoder.stdin.write(canvas.tobytes())
            if f in picked:shots.append(canvas.resize((480,480)))
    finally:
        encoder.stdin.close()
    if encoder.wait()!=0:raise RuntimeError('ffmpeg failed')
    sheet=Image.new('RGB',(1440,960),'white')
    for i,im in enumerate(shots):sheet.paste(im,((i%3)*480,(i//3)*480))
    sheet.save(run/'contact_sheet.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-print_format','json','-show_streams','-show_format',str(run/'proof.mp4')]))
    write_json(run/'video.json',{'frame_count':n,'fps':fps,'duration_s':n/fps,'ffprobe':probe,
                               'label':label,'source_sha256':sha256(run/'source.npz'),
                               'target_sha256':sha256(run/'target.npz'),'video_sha256':sha256(run/'proof.mp4'),
                               'contact_sheet_frames':sorted(picked),'command':command,'review_status':'pending visual inspection'})
