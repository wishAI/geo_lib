import numpy as np
import pytest
from landau import Robot
from validate import validate, semantics, segment_distance

@pytest.fixture(scope='module')
def robot():return Robot()

@pytest.fixture
def still(robot):
    n=15;q=np.zeros((n,len(robot.names)));base=np.repeat(np.eye(4)[None],n,axis=0)
    lowest=min(v[:,2].min() for _,v,_ in robot.vertices(robot.fk(q[0])))
    base[:,2,3]=-lowest
    return dict(q=q,base=base,base_quat_xyzw=np.tile([0.,0,0,1.],(n,1)),times=np.arange(n)/30,
                errors_m=np.zeros((n,20)),joint_names=np.array(robot.names))

def checks(report):return {v['check'] for v in report['violations']}

def test_static_fixture_passes_kinematic_only(robot,still):
    report=validate(still,robot)
    assert report['kinematic_pass'],report['violations']
    assert report['dynamic_feasibility']=='not tested'

@pytest.mark.parametrize('fault,expected',[
    ('nan','nonfinite'),('time','time_continuity'),('norm','quaternion_norm'),
    ('sign','quaternion_sign_discontinuity'),('joint','joint_limits'),('speed','joint_speed'),
    ('locked','locked_joint'),('floor','floor_penetration'),('slide','foot_sliding'),
    ('root','root_discontinuity'),('error','retarget_peak'),('order','joint_order')])
def test_invalid_motion_is_rejected(robot,still,fault,expected):
    if fault=='nan':still['q'][3,0]=np.nan
    elif fault=='time':still['times'][4]=still['times'][3]
    elif fault=='norm':still['base_quat_xyzw'][4]*=2
    elif fault=='sign':still['base_quat_xyzw'][4]*=-1
    elif fault=='joint':still['q'][3,0]=9
    elif fault=='speed':still['q'][3,0]=.5
    elif fault=='locked':still['q'][3,robot.locked[0]]=.02
    elif fault=='floor':still['base'][:,2,3]-=.03
    elif fault=='slide':still['base'][:,1,3]=np.arange(15)*.01
    elif fault=='root':still['base'][4,1,3]=.3
    elif fault=='error':still['errors_m'][5,2]=.2
    elif fault=='order':still['joint_names']=still['joint_names'][::-1]
    report=validate(still,robot)
    assert not report['kinematic_pass']
    assert expected in checks(report)

def test_acceleration_without_excess_speed(robot,still):
    still['q'][4,0]=.12
    assert 'joint_acceleration' in checks(validate(still,robot))

def test_semantic_valid_invalid_examples():
    names=['root_x','foot_l','foot_r','hand_l','hand_r','spine_03_x'];t=np.arange(121)/30
    p=np.zeros((len(t),len(names),3));yaw=np.zeros(len(t))
    assert semantics('idle',p,names,yaw,t)['status']=='passed'
    assert semantics('walk',p,names,yaw,t)['status']=='failed'
    p[:,:,1]=t[:,None]*.1;p[:,1,1]+=.03*np.sin(t*5)
    assert semantics('walk',p,names,yaw,t)['status']=='passed'
    p[:]=0;yaw=t*.4
    assert semantics('turn',p,names,yaw,t)['status']=='passed'
    assert semantics('turn',p,names,yaw*0,t)['status']=='failed'
    p[:,3,2]=.1;p[:,3,0]=.07*np.sin(t*6)
    assert semantics('wave',p,names,yaw*0,t)['status']=='passed'
    p[:,3,0]=0
    assert semantics('wave',p,names,yaw*0,t)['status']=='failed'

def test_segment_collision_crossing_parallel_disjoint():
    a=np.array([-1.,0,0]);b=-a;c=np.array([0.,-1,0]);d=-c
    assert segment_distance(a,b,c,d)<1e-8
    assert segment_distance(a,b,a+[0,1,0],b+[0,1,0])==pytest.approx(1)
    assert segment_distance(a,b,a+[0,0,2],b+[0,0,2])==pytest.approx(2)
