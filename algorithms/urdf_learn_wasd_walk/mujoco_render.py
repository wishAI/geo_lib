"""Render recorded MuJoCo states in a fresh process without importing Torch."""
import argparse
from pathlib import Path
import time
import imageio.v2 as imageio
import mujoco
import numpy as np
from algorithms.urdf_learn_wasd_walk.mujoco_backend import digest, write_json


def render(directory, *, azimuth=125., distance=2.1):
    out=Path(directory)
    start=time.perf_counter()
    states=np.load(out/'trajectory.npz')
    model=mujoco.MjModel.from_xml_path(str(out/'model.xml'))
    floor_extent=max(10., float(np.max(np.abs(states['qpos'][:,:2])))+5.)
    plane_display_overrides=[]
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id]==mujoco.mjtGeom.mjGEOM_PLANE:
            plane_display_overrides.append({'geom_id':geom_id,'original_size':model.geom_size[geom_id].tolist()})
            # Plane collisions are infinite; these two sizes bound only its display list.
            model.geom_size[geom_id,:2]=floor_extent
    # Rendering only: headlight follows the camera as the recorded character travels.
    # Physical model, recorded states and collision geometry are never changed.
    model.vis.headlight.active=1
    model.vis.headlight.ambient[:]=[.25,.25,.25]
    model.vis.headlight.diffuse[:]=[.65,.65,.65]
    data=mujoco.MjData(model)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[0,0,.43]
    camera.distance,camera.azimuth,camera.elevation=distance,azimuth,-12
    renderer=mujoco.Renderer(model,height=480,width=640)
    data.qpos[:]=states['qpos'][0]
    mujoco.mj_forward(model,data)
    renderer.update_scene(data,camera=camera)
    renderer.render()
    selected=[]
    render_retries=0
    from OpenGL import GL
    renderer_name=GL.glGetString(GL.GL_RENDERER).decode()
    with imageio.get_writer(out/'proof.mp4',fps=50,codec='libx264') as writer:
        for i,q in enumerate(states['qpos']):
            data.qpos[:]=q
            mujoco.mj_forward(model,data)
            camera.lookat[:2]=data.qpos[:2]
            renderer.update_scene(data,camera=camera)
            # Require visibility before encoding; retry only the SAME recorded state.
            # Never advance or alter physics to recover a rendering frame.
            for attempt in range(4):
                frame=renderer.render()
                GL.glFinish()
                rgb=frame.astype(float)
                mask=(rgb[:,:,0]>1.1*rgb[:,:,1]) & (rgb[:,:,1]>1.1*rgb[:,:,2]) & (rgb[:,:,0]>25)
                yy,xx=np.nonzero(mask)
                if len(xx)>100 and xx.min()>2 and xx.max()<637 and yy.min()>2 and yy.max()<477:
                    break
                render_retries+=1
                renderer.update_scene(data,camera=camera)
            else:
                raise RuntimeError(f'Character missing/cropped in proof frame {i}; refusing incomplete proof')
            writer.append_data(frame)
            if i in {0,len(states['qpos'])//2,len(states['qpos'])-1}:
                selected.append(frame.copy())
    renderer.close()
    imageio.imwrite(out/'contact_sheet.png',np.concatenate(selected,axis=1))
    result={'kind':'state_replay_of_exact_dynamics_trajectory','video_sha256':digest(out/'proof.mp4'),
            'trajectory_sha256':digest(out/'trajectory.npz'),'frames':len(states['qpos']),'fps':50,
            'visual_overrides':{'plane_display_half_extent_m':floor_extent,'camera_headlight_ambient':[.25,.25,.25],'camera_headlight_diffuse':[.65,.65,.65],'recorded_model_xml_or_trajectory_changed':False,'infinite_plane_display_overrides':plane_display_overrides},
            'render_wall_s':time.perf_counter()-start,'renderer_source_sha256':digest(__file__),
            'separate_process_without_torch':True,'same_state_render_retries':render_retries,'camera_azimuth':azimuth,'camera_distance':distance,'gl_renderer':renderer_name,'character_visible_every_frame':True}
    (out/'renderer_source.py').write_text(Path(__file__).read_text())
    write_json(out/'proof_metadata.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--azimuth',type=float,default=125.);p.add_argument('--distance',type=float,default=2.1);args=p.parse_args()
    render(args.directory,azimuth=args.azimuth,distance=args.distance)
