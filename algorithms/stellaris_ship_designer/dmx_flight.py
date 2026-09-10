"""Sample user-provided SFM DMX flight clips and retarget their shared pony rig.

Binary decoding uses the external MIT Blender Source Tools datamodel module.
DMX skeleton snapshots are posed, not bind poses. Animated root channels already
use Y-up/+Z-forward; child bone coordinates require a -90 degree local Z basis.
"""
from bisect import bisect_right
import hashlib
import importlib.util
from pathlib import Path

from mathutils import Matrix, Quaternion, Vector

CLIPS = {
    'moving_va': ('revamped flying fast.dmx', 8.975, 3.975, 97),
    'moving_genmaxx': ('flying_fast_xxgenmaxx_v1.dmx', 9.641666666666667, 4/3, 33),
    'moving_argodaemon': ('female_fly_cycle_fast_rev1.dmx', 5.0, 10.0, 241),
}


def matrix(position, xyzw):
    result = Quaternion((xyzw[3], *xyzw[:3])).normalized().to_matrix().to_4x4()
    result.translation = Vector(position)
    return result


def interpolate(layer, time, rotation=False):
    times, values = layer['times'], layer['values']
    index = max(0, min(len(times)-1, bisect_right(times, time)-1))
    value = values[index]
    if index == len(times)-1 or time <= times[0]:
        return list(value)
    fraction = (time-times[index])/(times[index+1]-times[index])
    if rotation:
        a, b = (Quaternion((v[3], *v[:3])).normalized() for v in (value, values[index+1]))
        q = a.slerp(b, fraction)
        return [q.x, q.y, q.z, q.w]
    return list(Vector(value).lerp(Vector(values[index+1]), fraction))


def streaming_tail(data, source_world, attachment, fraction, cycles, phase=0.0, lock_seamless=True):
    """Pose the existing skinned tail chain backward without altering its bind.

    Aim each original segment independently, preserving lengths and each bone's
    rest roll. This unfolds the hanging middle section while keeping the mesh's
    sculpted tip curl. Small delayed offsets provide a closed, authored follow-through.
    """
    from math import sin, pi
    bind = source_world(data, 0)
    names = ['Tail1', 'Tail2', 'Tail3']
    last = data['names'].index('Tail3')
    weighted = [(Vector(p), weights.get(last, 0)) for p, weights in zip(data['positions'], data['weights'])]
    tip = sum((point*weight for point, weight in weighted), Vector()) / sum(weight for _, weight in weighted)
    endpoints = [bind['Tail2'].translation, bind['Tail3'].translation, tip]
    position = attachment.translation.copy()
    result = {}
    for index, name in enumerate(names):
        span = endpoints[index] - bind[name].translation
        angle = 2*pi*(cycles*fraction + phase) - .55*index
        sway = sin(angle)
        lift = sin(angle+.4)
        if lock_seamless:
            start_angle = 2*pi*phase - .55*index
            end_angle = 2*pi*(cycles+phase) - .55*index
            sway += fraction*(sin(start_angle)-sin(end_angle))
            lift += fraction*(sin(start_angle+.4)-sin(end_angle+.4))
        direction = Vector((.025*(index+1)*sway, -.04-.055*index+.02*lift, -1)).normalized()
        rotation = span.normalized().rotation_difference(direction).to_matrix()
        transform = (rotation @ bind[name].to_3x3()).to_4x4()
        transform.translation = position
        result[name] = transform
        position = transform @ (bind[name].inverted() @ endpoints[index])
    return result


class Flight:
    def __init__(self, clip, wing_cycles=4.0, wing_phase=0.0, tail_cycles=4.0,
                 body_duration=4.0, body_speed=1.0, wing_speed=1.0,
                 tail_phase=0.0, lock_seamless=True):
        root = Path(__file__).resolve().parents[2]
        decoder = root/'helper_repos/blender_source_tools/datamodel.py'
        if not decoder.exists():
            raise FileNotFoundError('Install Blender Source Tools datamodel.py under helper_repos/blender_source_tools (MIT).')
        spec = importlib.util.spec_from_file_location('heart_dmx_datamodel', decoder)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        filename, self.start, self.duration, source_frames = CLIPS[clip]
        self.frames = round(float(body_duration)*24)+1 if clip == 'moving_va' else source_frames
        self.path = Path(__file__).parent/'inputs/heart_animation/community_flights'/filename
        data = module.load(str(self.path))
        model = data.root['skeleton']
        animation = data.root['animationList']['animations'][0]
        self.bones, self.ids, self.channels = {}, {}, {}
        def visit(node, parent=None):
            transform = node['transform']
            self.bones[node.name] = {'parent': parent, 'position': list(transform['position']), 'orientation': list(transform['orientation'])}
            self.ids[str(transform.id)] = node.name
            for child in node.get('children', []):
                visit(child, node.name)
        visit(model)
        for channel in animation['channels']:
            target = channel.get('toElement')
            if target is None or str(target.id) not in self.ids:
                continue
            log = channel['log']
            layers = log['layers']
            if len(layers) != 1 or layers[0].get('compressed', b'') or layers[0].get('curvetypes', []):
                raise ValueError('Only explicit single-layer DMX transform tracks are supported')
            layer = layers[0]
            if not layer['times'] or len(layer['times']) != len(layer['values']):
                raise ValueError('Invalid DMX key arrays')
            self.channels[self.ids[str(target.id)], channel['toAttribute']] = {'times': list(map(float, layer['times'])), 'values': [list(v) for v in layer['values']]}
        self.clip = clip
        self.wing_cycles = float(wing_cycles)
        self.wing_phase = float(wing_phase)
        self.tail_cycles = float(tail_cycles)
        self.body_duration = float(body_duration) if clip == 'moving_va' else (self.frames-1)/24
        self.body_speed = float(body_speed) if clip == 'moving_va' else 1.0
        self.wing_speed = float(wing_speed) if clip == 'moving_va' else 1.0
        self.tail_phase = float(tail_phase)
        self.lock_seamless = bool(lock_seamless)
        self.root = model.name
        source_chain = 'Wing1 / Wing2 / Wing3' if clip == 'moving_va' else 'WingOpen / WingOpen0 / WingOpen02'
        self.report = {'file': filename, 'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(), 'model': model.get('modelName'), 'sourceFps': animation['frameRate'], 'sampleFps': 24, 'windowStart': self.start, 'windowDuration': self.duration, 'frames': self.frames, 'outputDuration': (self.frames-1)/24, 'playbackSpeedRatio': self.duration/((self.frames-1)/24), 'sourceBones': len(self.bones), 'transformChannels': len(self.channels), 'rootCoordinates': 'animated Y-up/+Z-forward; posed snapshot Z-up/+X-forward', 'neutralBindAvailable': False, 'wingMapping': f'{source_chain} rotation deltas on the three PonyLumen large-wing joints; target rest orientation and joint offsets are preserved'}
        self.report['wingSource'] = {'clip': clip, 'file': self.path.name, 'sha256': self.report['sha256'], 'cycles': self.wing_cycles if clip == 'moving_va' else 1, 'phase': self.wing_phase if clip == 'moving_va' else 0, 'speedMultiplier': self.wing_speed if clip == 'moving_va' else 1, 'attachment': 'Native main-wing chain applied rest-relative to PonyLumen large-wing transforms on Chest1'}
        self.report['liveSync'] = {'bodyDuration': self.body_duration, 'bodySpeed': self.body_speed, 'wingCycles': self.wing_cycles, 'wingSpeed': self.wing_speed, 'wingPhase': self.wing_phase, 'tailCycles': self.tail_cycles, 'tailPhase': self.tail_phase, 'lockSeamlessLoop': self.lock_seamless}

    def source_pose(self, time):
        local, world = {}, {}
        C = Matrix(((0,1,0,0), (0,0,1,0), (1,0,0,0), (0,0,0,1)))
        for name, bone in self.bones.items():
            values = {}
            for attr in ['position', 'orientation']:
                channel = self.channels.get((name, attr))
                values[attr] = interpolate(channel, time, attr == 'orientation') if channel else bone[attr]
            local[name] = matrix(values['position'], values['orientation'])
            # Root children fall back to snapshot space only when unanimated.
            if bone['parent'] == self.root and (name, 'orientation') not in self.channels:
                local[name] = C @ local[name]
            parent = bone['parent']
            world[name] = world[parent] @ local[name] if parent else local[name]
        return local, world

    def closed_source_pose(self, fraction):
        # Sample the chosen complete period. A tiny endpoint correction makes
        # every channel continuous without dropping asynchronous wing cycles.
        local, source = self.source_pose(self.start + fraction*self.duration)
        start_local, _ = self.source_pose(self.start)
        end_local, _ = self.source_pose(self.start+self.duration)
        for name, mat in local.items():
            qa = start_local[name].to_quaternion()
            qb = end_local[name].to_quaternion()
            correction = Quaternion().slerp(qb.rotation_difference(qa), fraction)
            rotation = mat.to_quaternion() @ correction
            position = mat.translation + fraction*(start_local[name].translation-end_local[name].translation)
            local[name] = rotation.to_matrix().to_4x4()
            local[name].translation = position
        source = {}
        for name, bone in self.bones.items():
            parent = bone['parent']
            source[name] = source[parent] @ local[name] if parent else local[name]
        return local, source

    def loop_pose(self, phase, cycles, fraction):
        """Sample any number of source cycles, optionally correcting the output seam."""
        current = (phase + cycles*fraction) % 1
        local, _ = self.closed_source_pose(current)
        if self.lock_seamless:
            start_local, _ = self.closed_source_pose(phase % 1)
            end_local, _ = self.closed_source_pose((phase + cycles) % 1)
            for name, mat in local.items():
                qa, qb = start_local[name].to_quaternion(), end_local[name].to_quaternion()
                correction = Quaternion().slerp(qb.rotation_difference(qa), fraction)
                rotation = mat.to_quaternion() @ correction
                position = mat.translation + fraction*(start_local[name].translation-end_local[name].translation)
                local[name] = rotation.to_matrix().to_4x4()
                local[name].translation = position
        world = {}
        for name, bone in self.bones.items():
            parent = bone['parent']
            world[name] = world[parent] @ local[name] if parent else local[name]
        return local, world

    def pose(self, fraction, sources, source_world, attachments):
        from math import pi
        nominal_duration = (CLIPS[self.clip][3]-1)/24
        body_cycles = self.body_speed*self.body_duration/nominal_duration if self.clip == 'moving_va' else 1
        local, source = self.loop_pose(0, body_cycles, fraction)
        # A's source contains seven native wing beats in its four-second
        # window.  The editor exposes beat count, so divide by seven when
        # sampling that complete source period.  This keeps the default motion
        # native while still allowing independent cycle, speed, and phase edits.
        if self.clip == 'moving_va':
            wing_phase = self.wing_phase / 7
            wing_cycles = self.wing_cycles * self.wing_speed / 7
            wing_local = self.loop_pose(wing_phase, wing_cycles, fraction)[0]
            wing_reference = self.closed_source_pose(wing_phase % 1)[0]
        else:
            wing_local = local
            wing_reference = self.closed_source_pose(0)[0]
        J = Matrix.Rotation(-pi/2, 4, 'Z')
        Ji = J.inverted()
        body = sources['pony']
        self.report['directBodyBones'] = [name for name in body['names'] if name in local and name != 'LeftEar' and (name != 'Tail1' or self.clip == 'moving_genmaxx')]
        self.report['snapshotFallbackBodyBones'] = [name for name in self.report['directBodyBones'] if (name, 'orientation') not in self.channels]
        self.report['targetRestMotion'] = ['left ear (source snapshot basis is not mirrored)', 'mane jiggle chains', 'facial expression and eye controls']
        self.report['tailMotion'] = 'Authored backward flight pose and delayed follow-through using all three original skinned tail joints; original mesh and bind weights preserved.'
        self.report['wingBonePairs'] = {}
        world = {}
        for idx, name in enumerate(body['names']):
            parent = body['parents'][idx]
            raw = body['local_matrices_column_major'][idx][0]
            target_local = Matrix([raw[i:i+4] for i in range(0,16,4)]).transposed()
            if name == 'vn_pony_reference':
                world[name] = Matrix.Identity(4)
                continue
            if name in local and name not in {'Tail1', 'LeftEar'}:
                # Use the Pelvis transform relative to the source root, not its
                # world transform. C's root carries a constant ~158-degree roll
                # plus a position around x=150/z=244; both are scene placement,
                # not character motion. A/B roots are identity, so this keeps
                # their existing pose while making C share the same ship frame.
                target_local = (local[name] @ Ji) if name == 'Pelvis' else J @ local[name] @ Ji
                # Keep Heart anchored and preserve every target joint offset so
                # all clips remain visible and usable as in-place ship loops.
                target_local.translation = Vector(raw[12:15])
            elif name == 'Tail1' and self.clip == 'moving_genmaxx':
                target_local = J @ local[name] @ Ji
                target_local.translation = Vector(raw[12:15])
            world[name] = world[body['names'][parent]] @ target_local
        for part, data in sources.items():
            if part == 'pony':
                continue
            part_world = source_world(data, 0, world[attachments[part]])
            if part == 'tail':
                part_world = streaming_tail(data, source_world, world[attachments[part]], fraction, self.tail_cycles if self.clip == 'moving_va' else 1, self.tail_phase, self.lock_seamless)
            if 'Wing' in part:
                side = 'Left' if part == 'leftWing' else 'Right'
                target_names = [side+'WingOpen', side+'WingOpen1', side+'WingOpen2']
                source_names = ([side+'Wing1', side+'Wing2', side+'Wing3'] if self.clip == 'moving_va'
                                else [side+'WingOpen', side+'WingOpen0', side+'WingOpen02'])
                self.report['wingBonePairs'].update({part+'_'+tn: sn for tn, sn in zip(target_names, source_names)})
                parent_world = world['Chest1']
                for tn, sn in zip(target_names, source_names):
                    idx = data['names'].index(tn)
                    raw = data['local_matrices_column_major'][idx][0]
                    target_rest = Matrix([raw[i:i+4] for i in range(0,16,4)]).transposed()
                    # Transfer the animation in the bone's own rest frame.
                    # Parent-frame premultiplication exaggerates mirrored basis
                    # differences and makes the left large wing fold inward.
                    source_delta = wing_reference[sn].inverted() @ wing_local[sn]
                    transform = target_rest @ (J @ source_delta @ Ji)
                    transform.translation = Vector(raw[12:15])
                    parent_world = parent_world @ transform
                    part_world[tn] = parent_world
            world.update({part+'_'+name: mat for name, mat in part_world.items()})
        return world
