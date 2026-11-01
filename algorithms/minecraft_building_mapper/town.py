"""TK2-only full-column extraction and inspectable, heuristic building recognition.

Raw chunks and all derived caches stay beside the remote world. No save writes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from scipy import ndimage as ndi
from PIL import Image

from .world import AnvilWorld, Block, read_level_metadata
from .catalog import Catalog, cover_visible_legacy_catalog, write_catalog
from .legacy_visuals import vanilla_visual
from .render import TextureRenderer

NATURAL_IDS = {0,1,2,3,7,8,9,10,11,12,13,14,15,16,17,18,21,31,32,37,38,39,40,56,59,60,73,74,78,79,80,81,82,83,86,87,88,99,100,103,104,105,106,110,111,115,121,127,129,141,142,153}
BUILT_IDS = {4,5,20,22,24,25,26,35,41,42,43,44,45,46,47,48,49,50,53,54,58,61,62,63,64,67,68,71,84,85,89,91,92,95,96,98,101,102,107,108,109,112,113,114,116,117,118,120,123,124,125,126,128,130,133,134,135,136,138,139,145,146,151,152,154,155,156}
ROAD_IDS = {4,5,13,24,35,43,44,98,125,126,155}


def unpack_chunk(chunk):
    """Dense y,z,x uint16 array: (12-bit legacy id << 4) | metadata."""
    out = np.zeros((256,16,16),dtype=np.uint16)
    if chunk is None: return out
    if chunk.modern: raise ValueError('Town extractor currently requires the Bilicraft 1.5.2 legacy save')
    for sy,(blocks,metadata,add) in chunk.legacy.items():
        if not 0<=sy<16: raise ValueError('Legacy section outside 0..255')
        ids=np.frombuffer(blocks,dtype=np.uint8).astype(np.uint16)
        def nibbles(raw):
            b=np.frombuffer(raw,dtype=np.uint8)
            v=np.empty(4096,dtype=np.uint16); v[0::2]=b&15; v[1::2]=b>>4
            return v
        if add is not None: ids |= nibbles(add)<<8
        codes=(ids<<4)|nibbles(metadata); codes[ids==0]=0
        out[sy*16:sy*16+16]=codes.reshape(16,16,16)
    return out


def state_record(code,catalog):
    bid,meta=divmod(int(code),16); key=f'legacy:{bid}:{meta}'
    entry=catalog.entry(key) or {}
    visual=entry.get('visual') or vanilla_visual(bid,meta)
    if 4084<=bid<=4089 and not entry.get('visual'): visual={'geometry':'alpha_cube','height':1}
    if bid==4092 and not entry.get('visual'): visual={'geometry':'cross','height':.8}
    name=str(entry.get('name',key))
    return {'code':int(code),'key':key,'name':name,'visual':visual,'resolution':entry.get('resolution','unknown'),'texture':entry.get('texture'),'renderAsAir':bool(entry.get('renderAsAir'))}


def recognize(height, ground, built_count, roof_height, roof_code, valid, bounds):
    """Separate elevated construction using ground-relative height and road gaps.

    This is uncalibrated candidate detection, never a claim of named semantics.
    """
    ground=ndi.median_filter(ground,size=9)
    rise=roof_height.astype(np.int16)-ground
    construction=(built_count>=3)&(rise>=3)&valid
    # A path must be low, built, and have several neighbouring path columns.
    low=np.isin(roof_code>>4,list(ROAD_IDS))&(abs(rise)<=2)&valid
    roads=low&(ndi.uniform_filter(low.astype(float),size=3)>=0.45)
    mask=ndi.binary_closing(construction,iterations=2)&~roads&valid
    mask=ndi.binary_fill_holes(mask)&valid
    labels,n=ndi.label(mask)
    objects=[]; result=np.zeros_like(labels,dtype=np.uint16)
    slices=ndi.find_objects(labels)
    for label,sl in enumerate(slices,1):
        if sl is None: continue
        local=labels[sl]==label; area=int(local.sum())
        if area<20: continue
        zz,xx=sl; w=xx.stop-xx.start; d=zz.stop-zz.start
        if min(w,d)<3: continue
        evidence=int(np.count_nonzero(construction[sl]&local))
        roof=int(np.percentile(roof_height[sl][local],90)); floor=int(np.median(ground[sl][local]))
        if roof-floor<3: continue
        ident=len(objects)+1; result[sl][local]=ident
        codes,counts=np.unique(roof_code[sl][local],return_counts=True)
        materials=sorted(zip(codes.tolist(),counts.tolist()),key=lambda x:-x[1])[:8]
        ez=slice(max(0,zz.start-3),min(height.shape[0],zz.stop+3)); ex=slice(max(0,xx.start-3),min(height.shape[1],xx.stop+3))
        extended=labels[ez,ex]==label
        ring=ndi.binary_dilation(extended,iterations=3)&~extended
        road_contact=int(np.count_nonzero(roads[ez,ex]&ring))
        score=round(min(.95,.35+.35*evidence/area+.15*min(1,(roof-floor)/10)+(.1 if road_contact else 0)),3)
        objects.append({'id':f'building-{ident:04d}','label':f'Building {ident:03d}','index':ident,
          'kind':'building candidate','confidence':score,'confidenceMeaning':'heuristic score; not calibrated probability',
          'bounds':{'x0':bounds['x0']+xx.start,'z0':bounds['z0']+zz.start,'x1':bounds['x0']+xx.stop,'z1':bounds['z0']+zz.stop,'y0':floor,'y1':roof+1},
          'footprintArea':area,'constructionColumns':evidence,'height':roof-floor+1,'roadContactColumns':road_contact,
          'roofMaterials':[{'code':c,'columns':n} for c,n in materials],
          'review':(['large connected complex; check for merged buildings'] if area>5000 else [])+(['selection boundary clips footprint'] if xx.start==0 or zz.start==0 or xx.stop==height.shape[1] or zz.stop==height.shape[0] else [])})
    return objects,result,roads,construction


def extract(world_path,catalog_path,region,output):
    if platform.system()=='Darwin': raise RuntimeError('Save extraction is TK2-only; use the remote command')
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'town.json').exists(): raise ValueError('Use a new output directory to preserve the previous extraction')
    started=time.time(); catalog=Catalog(catalog_path)
    raw=region['world']; b={'x0':math.floor(raw['x0']),'z0':math.floor(raw['z0']),'x1':math.ceil(raw['x1']),'z1':math.ceil(raw['z1'])}
    width=b['x1']-b['x0']; depth=b['z1']-b['z0']
    if width<=0 or depth<=0 or width*depth>4_000_000: raise ValueError('Region must contain 1..4 million columns')
    shape=(depth,width); height=np.zeros(shape,np.int16); ground=height.copy(); roof=height.copy()
    top=np.zeros(shape,np.uint16); roof_code=top.copy(); built_count=np.zeros(shape,np.uint8); valid=np.zeros(shape,bool)
    counts=Counter(); errors=[]; tiles=[]; chunks=0; hidden=0
    used=np.zeros(65536,np.int64); (output/'chunks').mkdir(exist_ok=True)
    lut=np.zeros(65536,bool); natural=np.zeros(65536,bool); render_air=np.zeros(65536,bool)
    for code in range(1,65536):
        bid=code>>4
        lut[code]=bid in BUILT_IDS
        natural[code]=bid in {1,2,3,7,12,13,80,82,87,88,110,121}
    for key,entry in catalog.entries.items():
        if not key.startswith('legacy:'): continue
        _,bid,meta=key.split(':'); bid=int(bid)
        states=range(16) if meta=='*' else [int(meta)]
        for m in states:
            code=bid*16+m
            render_air[code]=bool(entry.get('renderAsAir'))
            name=str(entry.get('name','')).lower()
            if bid>158 and any(t in name for t in ('brick','plank','concrete','marble','roof','tile','玻璃','砖','板','混凝土')): lut[code]=True
    yy=np.arange(256,dtype=np.int16)[:,None,None]
    with AnvilWorld(world_path) as world:
      level_metadata=read_level_metadata(world.fs)
      for cz in range(b['z0']//16,(b['z1']-1)//16+1):
        for cx in range(b['x0']//16,(b['x1']-1)//16+1):
          try: chunk=world.chunk(cx,cz); a=unpack_chunk(chunk)
          except Exception as e: errors.append({'x':cx,'z':cz,'error':str(e)}); continue
          if chunk is None: continue
          chunks+=1
          np.savez_compressed(output/'chunks'/f'{cx}.{cz}.npz',blocks=a,biomes=np.asarray(chunk.legacy_biomes,dtype=np.uint8))
          ax0=max(b['x0'],cx*16); az0=max(b['z0'],cz*16); ax1=min(b['x1'],cx*16+16); az1=min(b['z1'],cz*16+16)
          src=(slice(None),slice(az0-cz*16,az1-cz*16),slice(ax0-cx*16,ax1-cx*16)); dst=(slice(az0-b['z0'],az1-b['z0']),slice(ax0-b['x0'],ax1-b['x0']))
          v=a[src]; used+=np.bincount(v.ravel(),minlength=65536)
          hidden+=int(render_air[v].sum()); v=v.copy(); v[render_air[v]]=0
          h=np.max(np.where(v>0,yy,-1),axis=0); nat=np.max(np.where(natural[v],yy,-1),axis=0)
          rh=np.max(np.where(lut[v],yy,-1),axis=0)
          height[dst]=h; ground[dst]=np.where(nat>=0,nat,h); roof[dst]=rh
          top[dst]=np.take_along_axis(v,np.maximum(h,0)[None],axis=0)[0]
          roof_code[dst]=np.where(rh>=0,np.take_along_axis(v,np.maximum(rh,0)[None],axis=0)[0],0)
          built_count[dst]=np.minimum(255,np.sum(lut[v],axis=0)); valid[dst]=True
        if cz%10==0: print(json.dumps({'stage':'extract','chunkRow':cz,'chunks':chunks,'seconds':round(time.time()-started)}),flush=True)
    # Only genuinely observed top states trigger the existing audited fallback resolver.
    top_codes,top_counts=np.unique(top[valid],return_counts=True)
    inventory={'schema':'geo.minecraft-town-top-inventory/v1','surfaceStates':{f'legacy:{int(c)>>4}:{int(c)&15}':{'topColumns':int(n),'surfaceContributions':int(n)} for c,n in zip(top_codes,top_counts) if c}}
    (output/'top_inventory.json').write_text(json.dumps(inventory,indent=2))
    covered=cover_visible_legacy_catalog(catalog.data,inventory)
    catalog_path=output/'render_catalog.json';write_catalog(covered,catalog_path);catalog=Catalog(catalog_path)
    states=[state_record(c,catalog)|{'blocks':int(used[c])} for c in np.flatnonzero(used) if c]
    objects,labels,roads,construction=recognize(height,ground,built_count,roof,roof_code,valid,b)
    np.savez_compressed(output/'features.npz',height=height,ground=ground,top=top,roof=roof,roof_code=roof_code,built_count=built_count,valid=valid,labels=labels,roads=roads,construction=construction)
    # Compact satellite overview from actual block textures. Detail meshing uses full texture pixels.
    renderer=TextureRenderer(catalog,16); rgb=np.zeros((65536,3),np.uint8)
    for s in states:
        bid,meta=divmod(s['code'],16); image=renderer._texture(Block(s['key'],0,bid,meta),None)
        ar=np.asarray(image); alpha=ar[:,:,3]/255; denom=max(alpha.sum(),1)
        color=(ar[:,:,:3]*alpha[:,:,None]).sum((0,1))/denom
        tint=s['visual'].get('tint')
        if tint in {'grass','foliage','pine','birch'}: color*=np.array([.52,.78,.32])
        if tint=='water': color*=np.array([.3,.55,.95])
        rgb[s['code']]=color.astype(np.uint8)
    shade=np.clip(1+(height-np.roll(height,1,axis=0))*.045,.65,1.15)
    satellite=np.clip(rgb[top]*shade[:,:,None],0,255).astype(np.uint8); satellite[~valid]=[25,32,40]
    Image.fromarray(satellite).save(output/'satellite.png')
    for name,mask,color in [('roads',roads,[245,190,73]),('construction',construction,[103,222,214])]:
        overlay=np.zeros((*shape,4),np.uint8); overlay[mask]=color+[210]; Image.fromarray(overlay).save(output/f'{name}.png')
    colors=np.random.default_rng(0).integers(70,240,(len(objects)+1,3),dtype=np.uint8); colors[0]=0
    overlay=np.concatenate((colors[labels],np.where(labels>0,170,0).astype(np.uint8)[:,:,None]),axis=2); Image.fromarray(overlay).save(output/'buildings.png')
    overview={'step':4,'width':math.ceil(width/4),'depth':math.ceil(depth/4),'heights':height[::4,::4].ravel().tolist()}
    (output/'terrain.json').write_text(json.dumps(overview,separators=(',',':')))
    report={'schema':'geo.minecraft-town/v1','id':region['id'],'name':region['name'],'mapId':region['mapId'],'bounds':b,'sourceSelection':region,
      'source':{'world':str(world_path),'levelMetadata':level_metadata,'catalog':str(catalog_path),'catalogSha256':hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(),'readOnly':True,'runtime':'TK2'},
      'chunks':chunks,'missingColumns':int((~valid).sum()),'blockCount':int(used[1:].sum()),'nonRenderingBlocks':hidden,'errors':errors,'states':states,'buildings':objects,
      'recognition':{'method':'ground-relative construction columns, road barriers, morphological footprints','verifiedComplete':False,'parameters':{'minimumHeight':3,'minimumArea':20,'closingRadius':2},'limitations':['Connected buildings can merge; detached parts can split.','Trees, terrain, underground rooms and unusual mod materials can obscure footprints.','Names and usage require human review.']},
      'stages':[{'id':'source','label':'Saved box','status':'success'},{'id':'roads','label':'Roads & ground','status':'success'},{'id':'construction','label':'Construction evidence','status':'success'},{'id':'buildings','label':'Building candidates','status':'warning'}],
      'seconds':round(time.time()-started,2),'textureFailures':renderer.failures}
    (output/'town.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'buildings':len(objects),'blocks':report['blockCount'],'states':len(states),'errors':len(errors),'seconds':report['seconds']}),flush=True)
    return report


def export_buildings(root):
    """Write individual candidate packages on TK2, retaining mask and context.

    The rectangular volume is lossless within its recorded bounds; the mask is
    the inferred footprint, not an asserted per-block ownership classification.
    """
    if platform.system()=='Darwin': raise RuntimeError('Building packages remain on TK2')
    import base64
    root=Path(root); report=json.loads((root/'town.json').read_text()); b=report['bounds']
    out=root/'buildings';out.mkdir(exist_ok=True)
    with np.load(root/'features.npz') as features:
        labels=features['labels'];height=features['height']
    packages=[]
    def safe(value):
        if isinstance(value,(bytes,bytearray)):return {'encoding':'base64','data':base64.b64encode(value).decode()}
        if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
        if isinstance(value,list):return [safe(v) for v in value]
        return value
    with AnvilWorld(report['source']['world']) as world:
      for item in report['buildings']:
        q=item['bounds'];z0=q['z0']-b['z0'];x0=q['x0']-b['x0'];z1=q['z1']-b['z0'];x1=q['x1']-b['x0']
        ymin=max(0,q['y0']-4);ymax=min(256,int(height[z0:z1,x0:x1].max())+1)
        data=np.zeros((ymax-ymin,q['z1']-q['z0'],q['x1']-q['x0']),np.uint16);entities=[]
        for cz in range(q['z0']//16,(q['z1']-1)//16+1):
          for cx in range(q['x0']//16,(q['x1']-1)//16+1):
            path=root/'chunks'/f'{cx}.{cz}.npz'
            if not path.exists():continue
            ax0=max(q['x0'],cx*16);ax1=min(q['x1'],cx*16+16);az0=max(q['z0'],cz*16);az1=min(q['z1'],cz*16+16)
            with np.load(path) as f:data[:,az0-q['z0']:az1-q['z0'],ax0-q['x0']:ax1-q['x0']]=f['blocks'][ymin:ymax,az0-cz*16:az1-cz*16,ax0-cx*16:ax1-cx*16]
            region=world._region(cx//32,cz//32);raw=region.read_chunk(cx,cz) if region else None
            body=(raw or {}).get('Level',raw or {})
            for entity in body.get('TileEntities',[]):
                ex=entity.get('x',-99999999);ez=entity.get('z',-99999999);ey=entity.get('y',-99999999)
                if q['x0']<=ex<q['x1'] and q['z0']<=ez<q['z1'] and ymin<=ey<ymax:entities.append(safe(entity))
        footprint=labels[z0:z1,x0:x1]==item['index']
        target=out/(item['id']+'.npz');np.savez_compressed(target,blocks=data,footprint=footprint,origin=np.array([q['x0'],ymin,q['z0']],np.int32))
        metadata={'schema':'geo.minecraft-building-crop/v1','building':item,'bounds':q|{'y0':ymin,'y1':ymax},'arrayOrder':'y,z,x','stateEncoding':'legacy_id * 16 + metadata','nonAirBlocks':int(np.count_nonzero(data)),'tileEntities':entities,'scope':'Rectangular source-block crop with a separate inferred footprint mask. Includes contextual ground, plants and adjacent material; ownership is not asserted.','file':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        (out/(item['id']+'.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        packages.append({k:metadata[k] for k in ('file','bytes','sha256','nonAirBlocks')})
    result={'schema':'geo.minecraft-building-extracts/v1','runtime':'TK2','directory':str(out),'count':len(packages),'packages':packages}
    (root/'extracts.json').write_text(json.dumps(result,indent=2));print(json.dumps({'exported':len(packages),'bytes':sum(p['bytes'] for p in packages)}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--world',required=True); p.add_argument('--catalog',required=True); p.add_argument('--regions',required=True); p.add_argument('--region',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); regions=json.loads(Path(a.regions).read_text())['regions']; region=next(r for r in regions if r['id']==a.region)
    extract(a.world,a.catalog,region,a.output)
    export_buildings(a.output)

if __name__=='__main__': main()
