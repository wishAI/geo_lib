"""Audit six-direction visible block states without meshing or copying saves."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .catalog import Catalog,cover_visible_legacy_catalog,write_catalog
from .town import state_record


def cover(root):
    root=Path(root);r=json.loads((root/'town.json').read_text());catalog=Catalog(r['source']['catalog'])
    solid=np.zeros(65536,bool);hidden=np.zeros(65536,bool)
    for record in r['states']:
        code=record['code'];solid[code]=record['visual'].get('geometry')=='cube';hidden[code]=record['renderAsAir']
    def load(x,z):
        p=root/'chunks'/f'{x}.{z}.npz'
        if not p.exists():return np.zeros((256,16,16),np.uint16)
        with np.load(p) as f:a=f['blocks'];a[hidden[a]]=0;return a
    counts=np.zeros(65536,np.int64)
    paths=sorted((root/'chunks').glob('*.npz'))
    for i,p in enumerate(paths):
        x,z=map(int,p.stem.split('.'));a=load(x,z);v=~solid[a];exposed=np.zeros_like(v)
        exposed[:,:,1:]|=v[:,:,:-1];exposed[:,:,:-1]|=v[:,:,1:];exposed[:,1:,:]|=v[:,:-1,:];exposed[:,:-1,:]|=v[:,1:,:];exposed[1:]|=v[:-1];exposed[:-1]|=v[1:]
        exposed[:,:,0]|=~solid[load(x-1,z)[:,:,-1]];exposed[:,:,-1]|=~solid[load(x+1,z)[:,:,0]]
        exposed[:,0,:]|=~solid[load(x,z-1)[:,-1,:]];exposed[:,-1,:]|=~solid[load(x,z+1)[:,0,:]];exposed[-1]=True
        counts+=np.bincount(a[exposed&(a!=0)].ravel(),minlength=65536)
        if i%2000==0:print('surface audit',i,'/',len(paths),flush=True)
    observed={f'legacy:{int(c)>>4}:{int(c)&15}':{'surfaceContributions':int(counts[c]),'topColumns':0} for c in np.flatnonzero(counts)}
    inventory={'schema':'geo.minecraft-town-3d-surface-inventory/v1','definition':'Non-air blocks adjacent to at least one non-opaque neighbour across six directions, with cross-chunk halos. Contributions count blocks, not faces.','surfaceStates':observed}
    (root/'surface_inventory_3d.json').write_text(json.dumps(inventory,indent=2))
    write_catalog(cover_visible_legacy_catalog(catalog.data,inventory),root/'viewer_catalog.json')
    print('3D surface states',len(observed),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();cover(a.root)
