"""Retarget the supplied 60 Hz run by rest-space deformation matrices.

The target rig is never edited. Full source rotations are transferred through
the respective world rest frames, including split shin/forearm twist bones.
Source root translation is retained (this supplied clip is in place).
"""
from pathlib import Path
import hashlib
import json
import bpy
from mathutils import Matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/landau_v10'
MAP = {'root_x':'Hips', 'spine_01_x':'Spine', 'spine_02_x':'Spine1',
       'spine_03_x':'Spine2', 'neck_x':'Neck', 'head_x':'Head'}
for side, suffix in [('Left', 'l'), ('Right', 'r')]:
    for target, source in [('shoulder_', 'Shoulder'), ('arm_stretch_', 'Arm'),
        ('arm_twist_', 'Arm'), ('forearm_stretch_', 'ForeArm'),
        ('forearm_twist_', 'ForeArm'), ('hand_', 'Hand'),
        ('thigh_stretch_', 'UpLeg'), ('thigh_twist_', 'UpLeg'),
        ('leg_stretch_', 'Leg'), ('leg_twist_', 'Leg'),
        ('foot_', 'Foot'), ('toes_01_', 'ToeBase')]:
        MAP[target+suffix] = side+source


def retarget(source, target, scene):
    source_scene = next(s for s in bpy.data.scenes if source.name in s.objects)
    source_action = source.animation_data.action
    start, end = map(int, source_action.frame_range)
    scale = .79
    ground = .006648703012615442*scale
    conversion = Matrix.Translation((0,0,ground)) @ Matrix.Scale(scale,4)
    target.animation_data_clear()
    for pb in target.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
        pb.rotation_mode = 'QUATERNION'
    action = bpy.data.actions.new('Running')
    target.animation_data_create().action = action
    rest = {n: (source.matrix_world@source.data.bones[n].matrix_local).inverted()
            for n in set(MAP.values())}
    root_samples, errors = [], []
    scene.render.fps = source_scene.render.fps
    scene.frame_start, scene.frame_end = start, end
    old_window_scene = bpy.context.window.scene
    for frame in range(start, end+1):
        bpy.context.window.scene = source_scene
        source_scene.frame_set(frame)
        deformations = {n: conversion @ (source.matrix_world@source.pose.bones[n].matrix) @ inv @ conversion.inverted()
                        for n, inv in rest.items()}
        root_samples.append(list(source.matrix_world@source.pose.bones['Hips'].head))
        bpy.context.window.scene = scene
        scene.frame_set(frame)
        for pb in target.pose.bones:
            if pb.name not in MAP:
                continue
            desired = target.matrix_world.inverted() @ deformations[MAP[pb.name]] @ target.matrix_world @ pb.bone.matrix_local
            pb.matrix = desired
            bpy.context.view_layer.update()
            errors.append((pb.matrix.translation-desired.translation).length)
            pb.keyframe_insert('location', frame=frame, group=pb.name)
            pb.keyframe_insert('rotation_quaternion', frame=frame, group=pb.name)
    bpy.context.window.scene = old_window_scene
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for fc in bag.fcurves:
                    for k in fc.keyframe_points:
                        k.interpolation = 'LINEAR'
    action.use_fake_user = True
    target.animation_data.action = None
    for pb in target.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    scene.frame_set(0)
    bpy.context.view_layer.update()
    report = dict(name='Running', source='~/Downloads/landau_body.fbx',
        source_sha256=hashlib.sha256((Path.home()/'Downloads/landau_body.fbx').read_bytes()).hexdigest(),
        source_action=source_action.name, fps=scene.render.fps, frames=[start,end],
        duration_seconds=(end-start)/scene.render.fps, mapped_bones=MAP,
        method='World pose times inverse world rest; uniform 0.79 unit conversion; target rest frame preserved.',
        max_joint_matrix_error=max(errors), root_samples=root_samples,
        root_motion='Source retained; in-place clip, no invented forward translation.')
    (OUT/'running_retarget.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('root_samples','mapped_bones')}))
    return action, report
