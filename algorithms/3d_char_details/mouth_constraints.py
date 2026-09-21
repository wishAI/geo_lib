"""Portable NumPy mouth-key untangling. Coordinates are character model units.
Preserves every patch boundary coordinate; only x/z coordinates can change.
No scipy, live-file operations, or renderer-specific approximations.
"""
import itertools
from collections import Counter
import numpy as np

def _cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]

def exact_area_minima(basis,deltas,triangles):
    """Minimize each triangle's quadratic signed x/z area on [0,1]x[-1,1]^2.
    Enumerate corners, edge/face stationary points and the interior stationary
    point. Flat singular restrictions attain the same minimum on their boundary.
    Returns model-unit squared areas and the minimizing three parameter values.
    """
    t=triangles;p=basis[:,[0,2]];d=deltas[:,:,[0,2]]
    e=p[t[:,1]]-p[t[:,0]];f=p[t[:,2]]-p[t[:,0]]
    ed=d[:,t[:,1]]-d[:,t[:,0]];fd=d[:,t[:,2]]-d[:,t[:,0]]
    c=_cross(e,f);g=np.stack([_cross(ed[i],f)+_cross(e,fd[i]) for i in range(3)],axis=1)
    H=np.stack([np.stack([_cross(ed[i],fd[j])+_cross(ed[j],fd[i]) for j in range(3)],axis=1) for i in range(3)],axis=1)
    vals=np.full(len(t),np.inf);args=np.zeros((len(t),3));lo=np.array([0,-1,-1]);hi=np.ones(3)
    for active in itertools.product([-1,0,1],repeat=3):
        free=np.flatnonzero(np.array(active)==0);fixed=np.flatnonzero(np.array(active)!=0)
        s=np.zeros((len(t),3));valid=np.ones(len(t),dtype=bool)
        s[:,fixed]=np.array([lo[i] if active[i]<0 else hi[i] for i in fixed])
        if len(free):
            hf=H[:,free][:,:,free];rhs=-(g[:,free]+np.einsum('nij,nj->ni',H[:,free][:,:,fixed],s[:,fixed]))
            scale=np.max(abs(hf),axis=(1,2));det=np.linalg.det(hf/np.maximum(scale[:,None,None],1e-30));valid=abs(det)>1e-12
            s[np.ix_(valid,free)]=np.linalg.solve(hf[valid],rhs[valid][...,None])[...,0]
            valid&=np.all((s[:,free]>=lo[free]-1e-9)&(s[:,free]<=hi[free]+1e-9),axis=1)
        v=c+np.einsum('ni,ni->n',g,s)+.5*np.einsum('ni,nij,nj->n',s,H,s)
        keep=valid&(v<vals);vals[keep]=v[keep];args[keep]=s[keep]
    return vals,args

def untangle_mouth(basis,triangles,deltas,progress=print):
    """Return (Basis shift, corrected jaw/length/curvature deltas, audit).
    Inputs contain only the mouth patch, with triangles wound toward -Y.
    Add Basis shift to EVERY existing key, including Basis, then overwrite the
    three solved key positions with old Basis + shift + corrected delta.
    """
    original=np.asarray(deltas,dtype=float).copy();basis=np.asarray(basis,dtype=float);t=np.asarray(triangles,dtype=int);n=len(basis)
    b=basis[:,[0,2]]/.04;fields=original[:,:,[0,2]]/.04
    edges=Counter(tuple(sorted((int(a),int(c)))) for tri in t for a,c in zip(tri,np.roll(tri,-1)))
    fixed=np.unique([e for e,count in edges.items() if count==1]);free=np.ones(n,dtype=bool);free[fixed]=False;trifree=free[t]
    def areas(p):return _cross(p[t[:,1]]-p[t[:,0]],p[t[:,2]]-p[t[:,0]])
    aa=areas(b);assert np.all(aa>0),'Basis must already have consistent mouth topology'
    def good(coeffs,target):return all(np.all(areas(b+np.einsum('i,ijk->jk',cs,fields))>=target*.999) for cs in coeffs)
    def project(coeffs,target,limit,coarse=False):
        nonlocal fields
        rng=np.random.default_rng(47)
        for iteration in range(limit):
            order=range(len(coeffs)) if coarse else rng.permutation(len(coeffs))
            for state in order:
                cs=coeffs[state];c2=np.dot(cs,cs)
                if c2==0:continue
                p=b+np.einsum('i,ijk->jk',cs,fields);A=areas(p);ids=np.flatnonzero(A<target)
                if not len(ids):continue
                if coarse:
                    ti=t[ids];pt=p[ti];g=np.empty_like(pt)
                    for j in range(3):
                        prev=pt[:,(j+1)%3];nxt=pt[:,(j+2)%3];g[:,j,0]=prev[:,1]-nxt[:,1];g[:,j,1]=nxt[:,0]-prev[:,0]
                    g*=trifree[ids,:,None];denom=(g*g).sum(axis=(1,2))*c2
                    corr=g*((target[ids]-A[ids])/np.maximum(denom,1e-30))[:,None,None]
                    add=np.zeros((n,2));counts=np.zeros(n)
                    for j in range(3):np.add.at(add,ti[:,j],corr[:,j]);np.add.at(counts,ti[:,j],trifree[ids,j])
                    fields+=.8*cs[:,None,None]*(add/np.maximum(counts[:,None],1))[None]
                else:
                    for i in ids:
                        ti=t[i];pt=p[ti];ar=_cross(pt[1]-pt[0],pt[2]-pt[0])
                        if ar>=target[i]:continue
                        g=np.empty_like(pt)
                        for j in range(3):
                            prev=pt[(j+1)%3];nxt=pt[(j+2)%3];g[j]=[prev[1]-nxt[1],nxt[0]-prev[0]]
                        g*=trifree[i,:,None];denom=(g*g).sum()*c2
                        corr=g*((target[i]*1.1-ar)/max(denom,1e-30))*1.1
                        fields[:,ti]+=cs[:,None,None]*corr[None];p[ti]+=c2*corr
            interval=25 if coarse else 50
            if iteration%interval==0 and good(coeffs,target):return iteration+1
        return limit
    coeffs=np.array(list(itertools.product([0,.25,.5,.75,1],[-1,0,1],[-1,0,1])))
    target=np.maximum(aa*.02,1e-10)
    stages=[project(coeffs,target,3000,True),project(coeffs,target,10000)]
    # Small neutral interior freedom conditions the narrow lip corners. Keep
    # boundaries pinned and carry this baseline shift through every other key.
    fields=np.concatenate([fields,np.zeros((1,n,2))])
    coeffs=np.array(list(itertools.product([0,.25,.5,.75,1],[-1,-.5,0,.5,1],[-1,-.5,0,.5,1])))
    rrng=np.random.default_rng(409);coeffs=np.vstack([coeffs,np.column_stack([rrng.uniform(0,1,128),rrng.uniform(-1,1,128),rrng.uniform(-1,1,128)])]);coeffs=np.column_stack([coeffs,np.ones(len(coeffs))])
    target=np.maximum(aa*.05,(np.linalg.norm(b[t[:,1]]-b[t[:,0]],axis=1)+np.linalg.norm(b[t[:,2]]-b[t[:,0]],axis=1))*4e-6)
    stages.append(project(coeffs,target,3000))
    for refinement in range(8):
        out=np.concatenate([original.copy(),np.zeros((1,n,3))]);out[:,:,[0,2]]=fields*.04
        serialized_basis=(basis+out[3]).astype('f').astype(float)
        serialized_deltas=(basis[None]+out[3]+out[:3]).astype('f').astype(float)-serialized_basis
        minimum,args=exact_area_minima(serialized_basis,serialized_deltas,t)
        progress('Mouth continuum audit',refinement,int(sum(minimum<=0)),float(minimum.min()))
        if np.all(minimum>0):
            assert np.array_equal(out[3,fixed],np.zeros((len(fixed),3)))
            assert np.array_equal(out[:3,fixed],original[:,fixed])
            return out[3],out[:3],dict(continuous_box_inverted_triangles=0,minimum_projected_double_area=float(minimum.min()),float32_keys=True,boundary_error=0.0,max_basis_correction=float(abs(out[3]).max()),max_key_correction=float(abs(out[:3]-original).max()),solver_iterations=stages,random_seed=47)
        extra=args[minimum<target*.0016*2]
        coeffs=np.vstack([coeffs,np.column_stack([extra,np.ones(len(extra))])])
        stages.append(project(coeffs,target,1000))
    raise RuntimeError('Mouth untangling did not pass the continuous float32 triangle audit')
