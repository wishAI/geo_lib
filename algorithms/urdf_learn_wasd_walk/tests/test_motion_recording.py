import unittest

import numpy as np

from algorithms.urdf_learn_wasd_walk.motion_recording import ANGLE_STEP, pack_motion


def restore(meta, payload):
    columns = []
    for column in meta['columns']:
        if 'constant' in column:
            values = np.full(meta['frames'], column['constant'])
        else:
            dtype = {'f32':'<f4', 'u16':'<u2', 'u32':'<u4'}[column['dtype']]
            values = np.frombuffer(payload, dtype=dtype, count=meta['frames'],
                                   offset=column['byteOffset']).astype(float)
            values = values * column['scale'] + column['offset']
        columns.append(values)
    clock = meta['clock']
    times = (clock['start'] + np.arange(meta['frames']) * clock['step']
             if 'step' in clock else np.frombuffer(payload, dtype='<f8',
                count=meta['frames'], offset=clock['byteOffset']))
    return np.array(columns).T, times


class MotionRecordingTests(unittest.TestCase):
    def motion(self):
        times = np.arange(101) * .02
        q = np.zeros((len(times), 10));q[:, 3] = 1
        q[:, 1] = times * .13
        q[:, 7] = np.sin(times)
        q[:, 8] = .7 + np.linspace(0, ANGLE_STEP*.8, len(times))
        q[:, 9] = np.linspace(-4, 4, len(times))
        joints = [dict(name='joint'+str(i), qpos=i) for i in range(7, 10)]
        return q, times, joints

    def test_complete_frames_root_and_bounded_joint_error(self):
        q, times, joints = self.motion()
        meta, payload = pack_motion(q, times, joints)
        decoded, clock = restore(meta, payload)
        self.assertEqual(decoded.shape, q.shape)
        np.testing.assert_allclose(clock, times, atol=1e-8)
        np.testing.assert_allclose(decoded[:, :7], q[:, :7], atol=1e-7)
        self.assertLessEqual(np.max(np.abs(decoded[:, 7:]-q[:, 7:])), ANGLE_STEP/2+1e-12)
        self.assertIn('constant', meta['columns'][8])
        self.assertEqual(meta['columns'][9]['dtype'], 'u32')
        self.assertEqual(meta['byteLength'], len(payload))

    def test_irregular_timestamps_preserved_exactly(self):
        q, times, joints = self.motion();times[50:] += .003
        meta, payload = pack_motion(q, times, joints)
        _, clock = restore(meta, payload)
        np.testing.assert_array_equal(clock, times)

    def test_rejects_corrupt_trajectory(self):
        for corruption in ('quaternion', 'nonfinite', 'time'):
            q, times, joints = self.motion()
            if corruption == 'quaternion':q[10, 3] = .5
            elif corruption == 'nonfinite':q[10, 7] = np.nan
            else:times[10] = times[9]
            with self.assertRaises(ValueError):pack_motion(q, times, joints)
