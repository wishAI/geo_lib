"""Fixed world-view debugging and a clean animation of the actual Landau meshes."""
import json
import subprocess
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from landau import Robot
from retarget import load_source, source_foot_contacts
from state import sha256, write_json

W,H=960,960
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def arrow(draw, start, end, color, text, font):
    start=np.asarray(start);end=np.asarray(end);delta=end-start
    draw.line([tuple(start),tuple(end)],fill=color,width=3)
    length=np.linalg.norm(delta)
    if length>3:
        direction=delta/length;normal=np.array([-direction[1],direction[0]])
        draw.polygon([tuple(end),tuple(end-8*direction+4*normal),tuple(end-8*direction-4*normal)],fill=color)
    else:
        # Out-of-image-plane directions must not masquerade as a missing arrow.
        draw.ellipse([end[0]-4,end[1]-4,end[0]+4,end[1]+4],outline=color,width=2)
        text+=' (depth)'
    draw.text(tuple(end+np.array([5,-15])),text,font=font,fill=color)


def encoder_command(path, width, height, fps):
    return ['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
            '-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast',
            '-crf','24','-pix_fmt','yuv420p',str(path)]


def clean_frame(meshes, root, height, frame, n, fps, font):
    """Fixed world three-quarter camera; no facing-dependent camera rotation."""
    width,height_px=960,720
    canvas=Image.new('RGB',(width,height_px),'#f1f3f4');draw=ImageDraw.Draw(canvas)
    azimuth=np.pi/4;elevation=np.deg2rad(12)
    right=np.array([-np.cos(azimuth),np.sin(azimuth),0.])
    horizontal_depth=np.array([np.sin(azimuth),np.cos(azimuth),0.])
    up=np.array([0.,0.,np.cos(elevation)])-horizontal_depth*np.sin(elevation)
    depth=horizontal_depth*np.cos(elevation)+np.array([0.,0.,np.sin(elevation)])
    center=np.array([root[0],root[1],0.]);scale=575/height
    def project(points):
        p=np.asarray(points)-center
        return np.stack([width/2+p@right*scale,650-p@up*scale],axis=-1)
    for offset in np.arange(-2,2.01,.25):
        for axis in (0,1):
            points=np.array([[-2.,offset,0.],[2.,offset,0.]])
            if axis:points=points[:,[1,0,2]]
            points+=center
            draw.line([tuple(p) for p in project(points)],fill='#dce2e4',width=1)
    polygons=[];light=np.array([-.3,.5,.81]);light/=np.linalg.norm(light)
    for link,vertices,faces in meshes:
        triangles=vertices[faces];normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        # STL winding is not relied on for lighting or back-face rejection.
        lighting=.62+.38*np.abs(normals@light)
        palette=np.array([154,179,190]) if 'head' in link or 'ear' in link else np.array([104,142,163])
        colors=np.clip(lighting[:,None]*palette,0,255).astype(int)
        xy=project(vertices)
        for face,z,color in zip(faces,(triangles@depth).mean(1),colors):
            polygons.append((z,xy[face],tuple(color)))
    for _,points,color in sorted(polygons,key=lambda item:item[0]):
        draw.polygon([tuple(v) for v in points],fill=color)
    draw.text((26,20),'Landau animation',font=font,fill='#304857')
    draw.text((26,47),f'{frame/fps:.2f} / {n/fps:.2f} s',font=font,fill='#607782')
    return canvas


def render(run, label):
    r=Robot();src,skeleton=load_source(run/'source.npz')
    mapping=json.loads((run/'retarget.json').read_text())
    coordinate_matrix=np.array(mapping['source_coordinate_matrix'])
    source=src['posed_joints']@coordinate_matrix.T
    source_contacts=source_foot_contacts(src['foot_contacts']) if 'foot_contacts' in src else None
    with np.load(run/'target.npz') as f:data={k:f[k] for k in f.files}
    report=json.loads((run/'validation.json').read_text())
    q=data['q'];base=data['base'];times=data['times'];n=len(times);fps=round(1/np.median(np.diff(times)))
    byframe={i:[] for i in range(n)}
    for v in report['violations']:
        if v['frame']>=0 and v['check'] in ('floor_penetration','foot_sliding','self_intersection_proxy','quaternion_step','root_discontinuity','joint_continuity','joint_jitter','joint_limits'):
            byframe[v['frame']].append(v['check'])
    font=ImageFont.truetype(FONT,16);small=ImageFont.truetype(FONT,13)
    max_z=0.
    for f in range(n):
        max_z=max(max_z,max(float(v[:,2].max()) for _,v,_ in r.vertices(r.fk(q[f],base[f]))))
    target_height=max(1.2,max_z+.08)
    source_height=max(1.85,float(source[:,:,2].max())+.12)
    picked=set(np.linspace(0,n-1,6,dtype=int).tolist());shots=[];clean_shots=[]
    command=encoder_command(run/'proof.mp4',W,H,fps)
    clean_command=encoder_command(run/'clean_preview.mp4',960,720,fps)
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    clean_encoder=subprocess.Popen(clean_command,stdin=subprocess.PIPE)
    name_index={a:i for i,(a,_) in enumerate(skeleton)}
    edges=[(name_index[parent],i) for i,(_,parent) in enumerate(skeleton) if parent is not None]
    rest=r.fk(np.zeros(len(r.names)))
    root_local_forward=rest['root_x'][:3,:3].T@np.array([0.,-1.,0.])
    if 'global_rot_mats' in src:
        source_forward=np.einsum('ij,tjk,k->ti',coordinate_matrix,src['global_rot_mats'][:,name_index['Hips']],np.array([0.,0.,1.]))
        forward_method='recorded source C @ global_rot_mats[Hips] @ SOMA local +Z'
    else:
        lateral=source[:,name_index['LeftLeg']]-source[:,name_index['RightLeg']]
        source_forward=np.cross(lateral,np.array([0.,0.,1.]))
        source_forward/=np.maximum(np.linalg.norm(source_forward,axis=1,keepdims=True),1e-12)
        forward_method='fallback: cross(named left-minus-right hips, world up); source rotations unavailable'
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
                    draw.text((ox+12,oy+8),f'{kind} | world {"XZ (from -Y)" if col==0 else "YZ (from +X)"}',font=font,fill='#152530')
                    if row==0:
                        xy=project(source[f])
                        for a,b in edges:draw.line([tuple(xy[a]),tuple(xy[b])],fill='#366889',width=4)
                        for pt in xy:draw.ellipse([pt[0]-2,pt[1]-2,pt[0]+2,pt[1]+2],fill='#17394f')
                        for k,foot in enumerate(['LeftFoot','RightFoot']):
                            pxy=xy[name_index[foot]]
                            contact=source_contacts is not None and bool(source_contacts[f,k])
                            draw.ellipse([pxy[0]-5,pxy[1]-5,pxy[0]+5,pxy[1]+5],outline='#10964d' if contact else '#db8e22',width=2)
                            color='#076ba8' if k==0 else '#a63d82'
                            toe=xy[name_index['LeftToeBase' if k==0 else 'RightToeBase']]
                            arrow(draw,pxy,toe,color,'Left toe' if k==0 else 'Right toe',small)
                            draw.text(tuple(pxy+np.array([5,8+k*13])),foot,font=small,fill=color)
                        arrow(draw,project(root),project(root+.25*source_forward[f]),'#be491a','source forward',small)
                    else:
                        polygons=[]
                        for link,vertices,faces in meshes:
                            projected=project(vertices);depth_axis=1 if axis==0 else 0
                            depths=vertices[faces,depth_axis].mean(1)*(-1 if axis==0 else 1)
                            for face,depth in zip(faces,depths):
                                color=(83,127,149) if link.endswith('_l') else (153,170,181) if link.endswith('_r') else (120,139,150)
                                polygons.append((depth,projected[face],color))
                        for _,points,color in sorted(polygons,key=lambda v:v[0]):draw.polygon([tuple(v) for v in points],fill=color,outline=(70,89,102))
                        for side in ('l','r'):
                            pos=tf['foot_'+side][:3,3];pxy=project(pos)
                            bottom=min(v[:,2].min() for l,v,_ in meshes if l in ('foot_'+side,'toes_01_'+side))
                            draw.ellipse([pxy[0]-6,pxy[1]-6,pxy[0]+6,pxy[1]+6],outline='#e13136' if bottom<-.005 else '#10964d' if bottom<=.012 else '#db8e22',width=3)
                            color='#076ba8' if side=='l' else '#a63d82'
                            arrow(draw,pxy,project(tf['toes_01_'+side][:3,3]),color,'Left toe' if side=='l' else 'Right toe',small)
                            draw.text(tuple(pxy+np.array([5,8 if side=='l' else 21])), 'Left ankle' if side=='l' else 'Right ankle',font=small,fill=color)
                        forward=tf['root_x'][:3,:3]@root_local_forward
                        arrow(draw,project(root),project(root+.18*forward),'#be491a','body forward',small)
                    draw.text((ox+12,oy+382),f'root XYZ: {root[0]:+.2f}, {root[1]:+.2f}, {root[2]:+.2f} m',font=small,fill='#243a49')
            draw.text((12,7),label[:108],font=font,fill='white')
            draw.text((12,29),f'frame {f+1}/{n} | {times[f]:.3f}s / {n/fps:.3f}s | original timing {fps}fps | animation',font=font,fill='#b5cbd7')
            warning=', '.join(sorted(set(byframe[f]))) or 'Source motion and Landau animation'
            draw.text((12,51),warning[:116],font=small,fill='#ff9388' if byframe[f] else '#90d2a4')
            draw.text((12,945),'Blue: Left | purple: Right | orange arrow: anatomical forward | fixed world cameras follow root',font=small,fill='white')
            encoder.stdin.write(canvas.tobytes())
            clean=clean_frame(meshes,tf['root_x'][:3,3],target_height,f,n,fps,font)
            clean_encoder.stdin.write(clean.tobytes())
            if f in picked:shots.append(canvas.resize((480,480)))
            if f in picked:clean_shots.append(clean.resize((480,360)))
    finally:
        encoder.stdin.close()
        clean_encoder.stdin.close()
    codes=[encoder.wait(),clean_encoder.wait()]
    if any(codes):raise RuntimeError(f'ffmpeg failed: {codes}')
    sheet=Image.new('RGB',(1440,960),'white')
    for i,im in enumerate(shots):sheet.paste(im,((i%3)*480,(i//3)*480))
    sheet.save(run/'contact_sheet.png')
    clean_sheet=Image.new('RGB',(1440,720),'white')
    for i,im in enumerate(clean_shots):clean_sheet.paste(im,((i%3)*480,(i//3)*360))
    clean_sheet.save(run/'clean_contact_sheet.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-print_format','json','-show_streams','-show_format',str(run/'proof.mp4')]))
    write_json(run/'video.json',{'frame_count':n,'fps':fps,'duration_s':n/fps,'ffprobe':probe,
                               'label':label,'source_sha256':sha256(run/'source.npz'),
                               'target_sha256':sha256(run/'target.npz'),'video_sha256':sha256(run/'proof.mp4'),
                               'contact_sheet_frames':sorted(picked),'command':command,'review_status':'pending visual inspection',
                               'camera_note':'Fixed world XZ viewed from -Y and YZ viewed from +X, root-following translation only. These are not anatomical front/side labels.',
                               'source_forward_method':forward_method,'source_coordinate_matrix':coordinate_matrix.tolist(),
                               'target_forward_method':'current root_x R @ rest root_x R.T @ canonical native base -Y',
                               'foot_arrow_note':'Named ankle-to-toe pivots, not inferred mesh heel contact positions',
                               'clean_preview':'clean_preview.mp4','clean_contact_sheet':'clean_contact_sheet.png'})
    clean_probe=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-print_format','json','-show_streams','-show_format',str(run/'clean_preview.mp4')]))
    write_json(run/'clean_video.json',{'frame_count':n,'fps':fps,'duration_s':n/fps,'ffprobe':clean_probe,
                                     'target_sha256':sha256(run/'target.npz'),'video_sha256':sha256(run/'clean_preview.mp4'),
                                     'contact_sheet_frames':sorted(picked),'command':clean_command,
                                     'camera':'Fixed world +X/+Y three-quarter orthographic view, 45 degree azimuth, 12 degree elevation; translation follows root',
                                     'geometry':'All actual copied Landau URDF visual mesh triangles at every target frame',
                                     'review_status':'pending visual inspection'})
    # Existing video-first GUI contract. This is native geometry rendering, not a crop.
    shutil.copyfile(run/'clean_preview.mp4',run/'preview.mp4')
