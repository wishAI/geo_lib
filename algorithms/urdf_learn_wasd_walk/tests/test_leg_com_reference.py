import numpy as np
from scipy.spatial.transform import Rotation

from algorithms.urdf_learn_wasd_walk.leg_com_reference import (
    bounded_com, contact_wrench_map, footsteps, ComReference, filter_heading, normal_force_reference,
)


def test_heading_reference_matches_critical_damping_without_derivative_kick():
    position = velocity = 0.
    for _ in range(50):
        position, velocity = filter_heading(position, velocity, .4)
    np.testing.assert_allclose([position, velocity], [.4 * (1 - 3 * np.exp(-2)), 1.6 * np.exp(-2)], atol=1e-14)


def test_normal_force_deficit_lowers_reference_smoothly_and_boundedly():
    position, velocity = np.zeros(2), np.zeros(2)
    for index in range(150):
        position, velocity = normal_force_reference(position, velocity, [0., 400.], [88., 88.], body_weight=176.)
        assert -.001 <= position[0] < 0 < position[1] <= .001
        if index == 0:
            assert np.max(abs(position)) < .00002
    np.testing.assert_allclose(position, [-.001, .001], atol=1e-11)
    position, velocity = normal_force_reference(np.zeros(2), np.zeros(2), [31., 176.], [0., 176.], body_weight=176.)
    np.testing.assert_array_equal(position, [0., 0.])
    np.testing.assert_array_equal(velocity, [0., 0.])


def test_entry_bridge_matches_incoming_position_velocity_acceleration():
    feet = np.array([[.045, -.02, .02], [-.045, .02, .02]])
    initial = np.array([0., 0., .373])
    velocity, acceleration = np.array([.0002, -.0001, 0.]), np.array([.0001, .0002, 0.])
    plan = ComReference(initial_feet=feet, initial_com=initial, cop_offset=[.0027, 0.],
                        step=.04, cycle=1., horizon=8., initial_velocity=velocity,
                        initial_acceleration=acceleration)
    _, _, com, v, a = plan.sample(0)
    np.testing.assert_allclose(com, initial, atol=1e-15)
    np.testing.assert_allclose(v, velocity, atol=1e-15)
    np.testing.assert_allclose(a, acceleration, atol=1e-15)
    for time in np.linspace(0, 4, 101):
        _, weights, _, _, _ = plan.sample(time)
        assert np.all(weights >= 0) and np.isclose(weights.sum(), 1)
    end = round(4 / plan.dt)
    np.testing.assert_allclose(plan.com[end] - plan.height / 9.81 * plan.acceleration[end], plan.zmp[end], atol=1e-14)
    assert np.linalg.norm(plan.velocity[end + 1] - plan.velocity[end]) < .003


def test_bounded_com_preserves_uniform_translation():
    dt = .01
    time = np.arange(501) * dt
    zmp = np.c_[.1 + .02 * time, -.04 * time]
    com, velocity, acceleration = bounded_com(
        zmp, height=.37, dt=dt, initial=zmp[0], terminal=zmp[-1])
    np.testing.assert_allclose(com, zmp, atol=1e-12)
    np.testing.assert_allclose(velocity, np.tile([.02, -.04], (501, 1)), atol=1e-10)
    np.testing.assert_allclose(acceleration, 0, atol=1e-10)


def test_bounded_com_matches_discrete_pendulum_not_unstable_forward_integration():
    dt, height = .002, .37
    time = np.arange(4001) * dt
    frequency = 2 * np.pi
    zmp = np.c_[.04 * np.sin(frequency * time), -.04 * time]
    response = 1 / (1 + height / 9.81 * 4 * np.sin(frequency * dt / 2)**2 / dt**2)
    expected = zmp.copy()
    expected[:, 0] *= response
    com, _, acceleration = bounded_com(
        zmp, height=height, dt=dt, initial=expected[0], terminal=expected[-1])
    np.testing.assert_allclose(com, expected, atol=1e-11)
    np.testing.assert_allclose(com[1:-1] - height / 9.81 * np.diff(com, n=2, axis=0) / dt**2,
                               zmp[1:-1], atol=1e-10)
    np.testing.assert_allclose(com - height / 9.81 * acceleration, zmp, atol=1e-14)


def test_footsteps_preserve_initial_asymmetry_and_contact_continuity():
    initial = np.array([[.045, -.014, .02], [-.045, .027, .021]])
    def sample(t):
        return footsteps(t, initial_feet=initial, step=.04, cycle=1., lift=.02)
    np.testing.assert_array_equal(sample(4)[0], initial)
    for boundary in [4., 5., 6., 7.]:
        before, wb = sample(boundary - 1e-8)
        after, wa = sample(boundary + 1e-8)
        np.testing.assert_allclose(before, after, atol=1e-8)
        np.testing.assert_allclose(wb, wa, atol=2e-7)
    feet, weights = sample(4.5)
    np.testing.assert_array_equal(feet[0], initial[0])
    np.testing.assert_array_equal(weights, [1., 0.])
    assert abs(feet[1, 2] - initial[1, 2] - .02) < 1e-12


def test_contact_wrench_closes_rotated_free_root_in_single_and_double_support():
    position = np.array([.3, -.2, .4])
    rotation = Rotation.from_rotvec([.2, -.1, .7]).as_matrix()
    points = np.array([[.35, -.21, 0], [.25, -.18, 0]])
    jp, jr = [], []
    for point in points:
        arm = point - position
        cross = np.array([[0, -arm[2], arm[1]], [arm[2], 0, -arm[0]], [-arm[1], arm[0], 0]])
        jp.append(np.c_[np.eye(3), -cross @ rotation])
        jr.append(np.c_[np.zeros((3, 3)), rotation])
    root_force = np.array([3., -2., 176., .6, -.4, .15])
    for weights in ([1., 0.], [.3, .7]):
        mapping, force_map, moment_map = contact_wrench_map(
            position, rotation, points, weights, jp, jr)
        np.testing.assert_allclose(mapping @ root_force, root_force, atol=1e-12)
        force = force_map @ root_force
        moment = moment_map @ root_force
        world_torque = np.cross(np.array(weights) @ points - position, force) + moment
        np.testing.assert_allclose(world_torque, rotation @ root_force[3:], atol=1e-12)


def test_swing_vertical_acceleration_is_continuous_at_contact_boundaries():
    initial = np.array([[.045, -.02, .02], [-.045, .02, .02]])
    def height(t):
        return footsteps(t, initial_feet=initial, step=.04, cycle=1., lift=.02)[0][1, 2]
    dt = .0001
    for boundary in [4.1, 4.9]:
        velocity = (height(boundary + dt) - height(boundary - dt)) / (2 * dt)
        acceleration = (height(boundary + dt) - 2 * height(boundary) + height(boundary - dt)) / dt**2
        assert abs(velocity) < 1e-6
        assert abs(acceleration) < .001
