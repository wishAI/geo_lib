import numpy as np
import pytest
from quality import intervals,summary,stance_displacement
from contact_refine import stance_targets


def test_intervals_and_worst_frame_report():
    assert intervals([True,True,False,True])==[(0,1),(3,3)]
    assert intervals([False,False])==[]
    stats=summary([1,3,4,2],np.arange(4)/30,2.)
    assert stats['worst_frame']==2 and stats['worst_time_s']==pytest.approx(2/30)
    assert stats['flagged_intervals']==[{'first_frame':1,'last_frame':2,'start_s':1/30,'end_s':2/30}]
    assert stats['p95']==pytest.approx(3.85)
    assert summary([1,2],[0,1],mask=[False,False])['sample_count']==0


def test_stance_targets_preserve_intended_source_motion_and_swing():
    source=np.zeros((12,3));source[:,0]=np.arange(12)*.01
    target=source+[.2,.1,.3];target[:,2]+=np.sin(np.arange(12))*.02
    stance=np.array([False]+[True]*7+[False]*4)
    goal,weight=stance_targets(target,source,stance)
    assert np.allclose(goal,target)
    assert np.all(weight[~stance]==0)
    distorted=target.copy();distorted[1:8,0]+=np.linspace(0,.02,7)
    goal,weight=stance_targets(distorted,source,stance)
    assert np.allclose(np.diff(goal[1:8,:2],axis=0),np.diff(source[1:8,:2],axis=0))
    assert np.array_equal(goal[:,2],distorted[:,2])
    assert np.array_equal(goal[~stance],distorted[~stance])
    residual,raw,source_drift=stance_displacement(target,source,stance)
    assert residual.max()<1e-12 and raw.max()>0 and source_drift.max()>0
    residual,_,_=stance_displacement(distorted,source,stance)
    assert residual.max()==pytest.approx(.02)
