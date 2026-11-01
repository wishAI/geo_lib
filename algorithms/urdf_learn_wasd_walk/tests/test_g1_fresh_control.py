import unittest

from algorithms.urdf_learn_wasd_walk.g1_fresh_control import completed_swings


class GaitEvidenceTests(unittest.TestCase):
    def test_contact_chatter_and_unfinished_flight_are_not_steps(self):
        contact = [[True, True], [False, False], [True, False], [False, False]]
        feet = [[[0, 0, z], [0, 0, z]] for z in (0, .03, .03, .03)]
        self.assertEqual(completed_swings(contact, feet, .02), [0, 0])

    def test_completed_airborne_bout_requires_clearance_on_each_side(self):
        contact = [[True, True]] + [[False, False]] * 4 + [[True, True]]
        feet = [[[0, 0, z], [0, 0, z / 10]] for z in (0, .01, .03, .03, .01, 0)]
        self.assertEqual(completed_swings(contact, feet, .02), [1, 0])
