"""Measured rabbit mouth seam from original front-facing mesh depth samples.

Input x, z, depth has depth shape (len(z), len(x)); forward is negative y.
Returns symmetric source seam samples and diagnostics, without fitting a new
analytic mouth contour. No SciPy dependency.
"""
import numpy as np


def gaussian_filter(values, sigma_samples, axis=0):
    radius=max(1,int(np.ceil(3*sigma_samples)))
    offsets=np.arange(-radius,radius+1)
    kernel=np.exp(-.5*(offsets/sigma_samples)**2);kernel/=kernel.sum()
    pads=[(0,0)]*values.ndim;pads[axis]=(radius,radius)
    padded=np.pad(values,pads,mode='edge')
    return np.apply_along_axis(lambda line:np.convolve(line,kernel,mode='valid'),axis,padded)


def parabola_minimum(grid, values, i):
    if i<=0 or i>=len(grid)-1:return float(grid[i])
    denominator=values[i-1]-2*values[i]+values[i+1]
    offset=0 if denominator<=0 else .5*(values[i-1]-values[i+1])/denominator
    return float(grid[i]+np.clip(offset,-.5,.5)*(grid[1]-grid[0]))


def trace_mouth_seam(x,z,depth,half_width=.03):
    x,z,depth=np.asarray(x),np.asarray(z),np.asarray(depth)
    assert depth.shape==(len(z),len(x))
    assert np.max(abs(x+x[::-1]))<1e-10,'Requires symmetric x samples'
    # Mirror average only valid source samples; a missing nose ray must not
    # poison a whole profile. Fill any remaining holes only within its column.
    mirrored=depth[:,::-1];valid=np.isfinite(depth);other=np.isfinite(mirrored)
    count=valid.astype(int)+other.astype(int)
    symmetric=(np.where(valid,depth,0)+np.where(other,mirrored,0))/np.maximum(count,1)
    symmetric[count==0]=np.nan
    for i in range(len(x)):
        ok=np.isfinite(symmetric[:,i])
        if np.count_nonzero(ok)<3:raise ValueError('Insufficient valid source depth')
        symmetric[:,i]=np.interp(z,z[ok],symmetric[ok,i])
    smooth=gaussian_filter(symmetric,.00035/np.median(np.diff(z)),axis=0)
    slope=np.gradient(smooth,z,axis=0);curvature=np.gradient(slope,z,axis=0)
    selected=np.where(abs(x)<=half_width+1e-10)[0]
    band=np.where((z>=.622)&(z<=.636))[0]
    raw=[];transition=[];strength=[]
    for i in selected:
        steep=band[np.argmin(slope[band,i])]
        # Lower concave foot of upper-muzzle projection, immediately before
        # the strongest forward step. The projecting crest is above this.
        candidates=band[(z[band]>=z[steep]-.002)&(z[band]<=z[steep]-.00015)]
        if len(candidates)==0:raise ValueError('No lower-foot search interval')
        foot=candidates[np.argmin(curvature[candidates,i])]
        raw.append(parabola_minimum(z,curvature[:,i],foot))
        transition.append(z[steep]);strength.append(-curvature[foot,i])
    seam_x=x[selected];raw=np.asarray(raw)
    # Filter only sub-millimeter tessellation noise across x. At the center
    # this preserves the source junction between the two muzzle lobes.
    seam_z=gaussian_filter(raw,.00065/np.median(np.diff(x)))
    seam_z=(seam_z+seam_z[::-1])/2
    seam_y=np.array([np.interp(zz,z,symmetric[:,i]) for zz,i in zip(seam_z,selected)])
    report={'method':'Measured lower concave foot of strongest upper-muzzle depth transition',
            'depth_z_sigma':.00035,'seam_x_sigma':.00065,
            'maximum_trace_smoothing_delta':float(np.max(abs(seam_z-raw))),
            'landmarks':[{ 'x':float(xx),'z':float(np.interp(xx,seam_x,seam_z)),
                           'y':float(np.interp(xx,seam_x,seam_y))}
                         for xx in np.arange(0,half_width+.000001,.003)]}
    return {'x':seam_x,'z':seam_z,'y':seam_y,'raw_z':raw,
            'transition_z':np.asarray(transition),'concavity':np.asarray(strength)},report
