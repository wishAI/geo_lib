"""Physics-step training guards retain transient peaks and exclude nonfeet."""
import unittest
import tempfile
try:
    import warp as wp
except ImportError:
    wp=None


@unittest.skipIf(wp is None, 'Requires the pinned training environment')
class GaitAuditTests(unittest.TestCase):
    def test_contact_frame_and_transient_joint_peaks(self):
        import numpy as np
        with tempfile.TemporaryDirectory(prefix='landau-warp-test-') as cache:
            wp.config.kernel_cache_dir=cache
            self.check_audit(np)

    def check_audit(self,np):
        from algorithms.urdf_learn_wasd_walk.landau_gait_search import (
            record_joint_speeds, clear_support, sum_support, record_support, record_airborne,
        )
        with wp.ScopedDevice('cpu'):
            def array(values, dtype):
                return wp.array(values, dtype=dtype, device='cpu')
            speed=wp.zeros(2,device='cpu')
            dofs=array([1,2],wp.int32)
            velocity=array([[100.,-4.2,1.],[100.,1.,-2.]],wp.float32)
            wp.launch(record_joint_speeds,2,[velocity,dofs,2,speed],device='cpu')
            velocity.zero_()
            wp.launch(record_joint_speeds,2,[velocity,dofs,2,speed],device='cpu')
            np.testing.assert_allclose(speed.numpy(),[4.2,2.])
            # Ground/foot, ground/nonfoot, foot/foot, then an unused slot.
            geom=array([[0,1],[0,2],[1,1],[0,1]],wp.vec2i)
            frame=array(np.tile([[0.,.6,.8],[0.,.8,-.6],[1.,0.,0.]],(4,1,1)),wp.mat33)
            support=wp.zeros(2,device='cpu');peak=wp.zeros_like(support)
            nonfoot=wp.zeros(2,dtype=wp.int32,device='cpu')
            normal=wp.zeros((2,2),device='cpu')
            args=[array([3],wp.int32),geom,array([0,0,1,1],wp.int32),
                  array([[0,2,1],[3,4,5],[0,1,2],[3,4,5]],wp.int32),frame,
                  array([[10.,2.,3.,100.,0.,0.],[50.,0.,0.,100.,0.,0.]],wp.float32),
                  array([-1,0,-1],wp.int32),support,nonfoot,normal]
            wp.launch(clear_support,2,[support,normal],device='cpu')
            wp.launch(sum_support,4,args,device='cpu')
            wp.launch(record_support,2,[support,peak],device='cpu')
            np.testing.assert_allclose(support.numpy(),[6.2,0.])
            np.testing.assert_array_equal(nonfoot.numpy(),[1,0])
            np.testing.assert_allclose(normal.numpy(),[[10.,0.],[0.,0.]])
            current=wp.zeros(2,dtype=wp.int32,device='cpu')
            maximum=wp.zeros_like(current);total=wp.zeros_like(current)
            for _ in range(7):
                wp.launch(record_airborne,2,[normal,current,maximum,total],device='cpu')
            np.testing.assert_array_equal(maximum.numpy(),[0,7])
            np.testing.assert_array_equal(total.numpy(),[0,7])
            normal.assign(np.array([[0.,0.],[0.,1.]],dtype=np.float32))
            for _ in range(3):
                wp.launch(record_airborne,2,[normal,current,maximum,total],device='cpu')
            np.testing.assert_array_equal(current.numpy(),[3,0])
            np.testing.assert_array_equal(maximum.numpy(),[3,7])
            np.testing.assert_array_equal(total.numpy(),[3,7])
            wp.launch(clear_support,2,[support,normal],device='cpu')
            wp.launch(record_support,2,[support,peak],device='cpu')
            np.testing.assert_allclose(peak.numpy(),[6.2,0.])
