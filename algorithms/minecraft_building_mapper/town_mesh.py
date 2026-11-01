"""Chunk-batched exposed-face meshes; original atlas pixels and non-cube plants."""
from __future__ import annotations
from collections import Counter
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .catalog import Catalog
from .legacy_visuals import VANILLA_152_BIOMES, biome_colorizer_xy
from .render import TextureRenderer
from .world import Block
from .town import state_record

# Face order: -X,+X,-Y,+Y,-Z,+Z. Quads wind outwards.
FACES=np.array([
 [[0,0,0],[0,0,1],[0,1,1],[0,1,0]],
 [[1,0,1],[1,0,0],[1,1,0],[1,1,1]],
 [[0,0,1],[0,0,0],[1,0,0],[1,0,1]],
 [[0,1,0],[0,1,1],[1,1,1],[1,1,0]],
 [[1,0,0],[0,0,0],[0,1,0],[1,1,0]],
 [[0,0,1],[1,0,1],[1,1,1],[0,1,1]],
],dtype=np.float32)
LIGHT=np.array([.8,.9,.55,1,.75,.94],np.float32)
TRI=np.array([0,1,2,0,2,3]); UV=np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)


def boxes(visual, bid, meta, connections=(False,False,False,False)):
    g=visual.get('geometry','cube')
    if g=='slab': return [(0,.5 if meta&8 else 0,0,1,1 if meta&8 else .5,1)]
    if g=='stair':
        upper=bool(meta&4); base=(0,.5 if upper else 0,0,1,1 if upper else .5,1)
        xy=[(.5,0,1,1),(0,0,.5,1),(0,.5,1,1),(0,0,1,.5)][meta&3]
        return [base,(xy[0],0 if upper else .5,xy[1],xy[2],.5 if upper else 1,xy[3])]
    if g=='connected':
        thin=bid in {101,102}; r=.0625 if thin else .125
        result=[(.5-r,0,.5-r,.5+r,1 if thin else 1.5,.5+r)]
        for n,connected in enumerate(connections):
            if not connected: continue
            for y0,y1 in ([(0,1)] if thin else [(.375,.5625),(.75,.9375)]):
                result.append([(0,y0,.5-r,.5,y1,.5+r),(.5,y0,.5-r,1,y1,.5+r),(.5-r,y0,0,.5+r,y1,.5),(.5-r,y0,.5,.5+r,y1,1)][n])
        return result
    if g in {'line','cover'}: return [(0,0,0,1,float(visual.get('height',.025)) if g=='cover' else .025,1)]
    if bid==111: return [(.05,0,.05,.95,.016,.95)]
    if g=='point': return [(.4,0,.4,.6,.8,.6)]
    if g=='plane':
        if bid==96 and not meta&4: return [(0,.8125 if meta&8 else 0,0,1,1 if meta&8 else .1875,1)]
        direction=(meta&3) if bid in {64,71} else ((meta-2)&3)
        return [(0,0,0,1,1,.125),(.875,0,0,1,1,1),(0,0,.875,1,1,1),(0,0,0,.125,1,1)][direction:direction+1]
    if g=='fluid': return [(0,0,0,1,1 if meta>=8 else max(.15,1-(meta+1)/9),1)]
    if g=='partial': return [(0,0,0,1,float(visual.get('height',.75)),1)]
    return [(0,0,0,1,1,1)]


def side_member(bid,meta,top):
    # Face selection is separate from the existing overhead catalog.
    values={2:'grass_side.png',17:('tree_side.png','tree_spruce.png','tree_birch.png','tree_jungle.png')[meta&3],24:('sandstone_carved.png' if meta==1 else 'sandstone_smooth.png' if meta==2 else 'sandstone_side.png'),43:'stoneslab_side.png',44:'stoneslab_side.png',46:'tnt_side.png',47:'bookshelf.png',58:'workbench_side.png',61:'furnace_side.png',62:'furnace_side.png',81:'cactus_side.png',84:'musicBlock.png',86:'pumpkin_side.png',91:'pumpkin_side.png',103:'melon_side.png',110:'mycel_side.png',155:'quartzblock_side.png'}
    if bid in {43,44} and meta&7: return top
    return 'textures/blocks/'+values[bid] if bid in values else top


class Mesher:
    def __init__(self,root):
        self.root=Path(root); self.report=json.loads((self.root/'town.json').read_text())
        with np.load(self.root/'features.npz') as features:
            self.labels=features['labels'];surface_heights=features['height']
        maxima=np.full(len(self.report['buildings'])+1,-1,dtype=np.int16)
        np.maximum.at(maxima,self.labels.ravel(),surface_heights.ravel())
        for item in self.report['buildings']:
            top=int(maxima[item['index']])+1
            if top>item['bounds']['y0']:
                item['estimatedTopBeforeAudit']=item['bounds']['y1']
                item['bounds']['y1']=top;item['height']=top-item['bounds']['y0']
                item['heightEvidence']='Maximum saved block Y in the detected footprint, plus one for its top edge'
        tallest=max(self.report['buildings'],key=lambda item:item['bounds']['y1'])
        tallest['label']='Highest tower · '+tallest['id'].replace('building-','B')
        tallest['kind']='tower candidate'
        self.report['tallestBuildingId']=tallest['id']
        self.buildings={item['id']:item for item in self.report['buildings']}
        catalog_path=self.root/'viewer_catalog.json'
        self.catalog=Catalog(catalog_path if catalog_path.exists() else self.report['source']['catalog']); self.records={s['code']:state_record(s['code'],self.catalog)|{'blocks':s['blocks']} for s in self.report['states']}
        self.solid=np.zeros(65536,bool); self.hidden=np.zeros(65536,bool); self.sprites={}; self.sprite_images=[]; self.sprite_keys={}
        self.failures={}; self.approximations={}; self.unknown_states={}; self.biome_cache={}; self.tint_maps={}
        for name in self.catalog.data.get('colorizers',{}):
            try: self.tint_maps[name]=Image.open(io.BytesIO(self.catalog.colorizer_bytes(name))).convert('RGB')
            except Exception as e: self.failures['colorizer:'+name]=str(e)
        renderer=TextureRenderer(self.catalog,32)
        names=[set(s.names()) for s in self.catalog.sources]
        for code,record in self.records.items():
            visual=record['visual']; g=visual.get('geometry','cube'); self.solid[code]=g=='cube'; self.hidden[code]=record['renderAsAir']
            bid,meta=divmod(code,16); entry=self.catalog.entry(record['key']) or {}; tex=entry.get('texture',{})
            if record['resolution']=='unknown':self.unknown_states[record['key']]={'name':record['name'],'blocks':record['blocks']}
            top_image=renderer._texture(Block(record['key'],0,bid,meta),None)
            top=self.sprite(top_image)
            faces=[top]*6; member=tex.get('member','')
            side=side_member(bid,meta,member)
            side_proven=side==member and (bid<=158 and bid not in {23,25,26,29,33,34,54,63,64,68,71,92,116,117,118,120,130,138,140,144,145,146,151,154,158})
            if side!=member:
                match=next((i for i,n in enumerate(names) if side in n),None)
                if match is not None:
                    im=Image.open(io.BytesIO(self.catalog.sources[match].read(side))).convert('RGBA'); sideid=self.sprite(im.crop((0,0,im.width,im.width)).resize((32,32),Image.Resampling.NEAREST))
                    faces=[sideid,sideid,top,top,sideid,sideid]; side_proven=True
                else: self.failures[record['key']+':side']=side
            if bid in {2,110}:
                dirt='textures/blocks/dirt.png'; match=next((i for i,n in enumerate(names) if dirt in n),None)
                if match is not None: faces[2]=self.sprite(Image.open(io.BytesIO(self.catalog.sources[match].read(dirt))).convert('RGBA').resize((32,32),Image.Resampling.NEAREST))
            if not side_proven and g not in {'cross','line','cover','alpha_cube'}:
                self.approximations[record['key']]='Side appearance uses the catalog representative; custom renderer not reconstructed'
            if g in {'point','plane','partial','nonstandard'}:
                self.approximations[record['key']]='Simplified legacy/custom shape; block identity retained'
            self.sprites[code]=faces
        self.failures.update(renderer.failures)
        self.atlas_cells=math.ceil(math.sqrt(max(1,len(self.sprite_images)))); self.atlas_size=self.atlas_cells*34
        atlas=Image.new('RGBA',(self.atlas_size,self.atlas_size))
        for i,im in enumerate(self.sprite_images):
            x=(i%self.atlas_cells)*34;y=(i//self.atlas_cells)*34
            atlas.paste(im,(x+1,y+1)); atlas.paste(im.crop((0,0,32,1)),(x+1,y)); atlas.paste(im.crop((0,31,32,32)),(x+1,y+33)); atlas.paste(im.crop((0,0,1,32)),(x,y+1)); atlas.paste(im.crop((31,0,32,32)),(x+33,y+1))
        atlas.save(self.root/'atlas.png')
        (self.root/'render_coverage.json').write_text(json.dumps({'textureFailures':self.failures,'unknownStates':self.unknown_states,'approximatedStates':self.approximations,'atlasSprites':len(self.sprite_images),'policy':'Full legacy block identity preserved. Unknown textures are magenta; unsupported custom shapes are explicitly listed. Entity models and minepainter tile-entity microgeometry are not yet reconstructed.'},ensure_ascii=False,indent=2))

    def sprite(self,image):
        image=image.convert('RGBA').resize((32,32),Image.Resampling.NEAREST); key=image.tobytes()
        if key not in self.sprite_keys: self.sprite_keys[key]=len(self.sprite_images); self.sprite_images.append(image)
        return self.sprite_keys[key]

    def uv(self,sprite):
        x=(sprite%self.atlas_cells)*34+1.01;y=(sprite//self.atlas_cells)*34+1.01
        out=UV.copy();out[:,0]=(x+out[:,0]*31.98)/self.atlas_size;out[:,1]=1-(y+(1-out[:,1])*31.98)/self.atlas_size
        return out

    def tint(self,code,biome,face):
        bid=code>>4; kind=self.catalog.block_colorizer(self.records[code]['key']) or self.records[code]['visual'].get('tint')
        if bid in {2,110} and face!=3: return np.ones(3)
        if not kind: return np.ones(3)
        if kind=='lily_pad':
            c=self.catalog.data.get('fixedColors',{}).get('lily_pad',0x208030);return np.array([(c>>16)&255,(c>>8)&255,c&255])/255
        if kind=='stem':
            m=code&15;return np.array([min(m*32,255),max(255-m*8,0),min(m*4,255)])/255
        key=(kind,int(biome),code if kind=='stem' else 0)
        if key not in self.biome_cache:
            climate=self.catalog.data.get('legacyBiomes',{}).get(str(biome))
            if climate: t=float(climate.get('temperature',.7));r=float(climate.get('rainfall',.8))
            else: _,t,r,_=VANILLA_152_BIOMES.get(int(biome),('unknown',.7,.8,0xffffff))
            mapname=kind if kind in self.tint_maps else ('foliage' if kind in {'pine','birch','foliage'} else kind)
            image=self.tint_maps.get(mapname)
            color=image.getpixel(biome_colorizer_xy(t,r)) if image else {'water':(90,140,230),'grass':(130,190,85),'foliage':(110,180,80),'pine':(97,153,97),'birch':(128,167,85),'lily_pad':(32,128,48)}.get(kind,(255,255,255))
            if kind=='water':
                water=int(climate.get('waterMultiplier',0xffffff)) if climate else VANILLA_152_BIOMES.get(int(biome),('unknown',.7,.8,0xffffff))[3]
                color=((water>>16)&255,(water>>8)&255,water&255)
            self.biome_cache[key]=np.asarray(color,dtype=np.float32)/255
        return self.biome_cache[key]

    def load(self,cx,cz):
        path=self.root/'chunks'/f'{cx}.{cz}.npz'
        if not path.exists(): return np.zeros((256,16,16),np.uint16),np.ones((16,16),np.uint8)
        with np.load(path) as f:
            a=f['blocks'];biomes=f['biomes'];a[self.hidden[a]]=0
            return a,biomes.reshape(16,16) if biomes.size==256 else np.ones((16,16),np.uint8)

    def clipped(self,cx,cz,cut,building=None,clip_bounds=None):
        a,biomes=self.load(cx,cz);a[cut+1:]=0;b=self.report['bounds']
        xx=np.arange(16)+cx*16;zz=np.arange(16)+cz*16
        a[:,:,(xx<b['x0'])|(xx>=b['x1'])]=0;a[:,(zz<b['z0'])|(zz>=b['z1']),:]=0
        if clip_bounds is not None:
            q=clip_bounds
            a[:,:,(xx<q['x0'])|(xx>=q['x1'])]=0;a[:,(zz<q['z0'])|(zz>=q['z1']),:]=0
            a[:q['y0']]=0;a[q['y1']:]=0
        elif building is not None:
            item=self.buildings[building];q=item['bounds'];mask=np.zeros((16,16),bool)
            ax0=max(cx*16,b['x0']);ax1=min(cx*16+16,b['x1']);az0=max(cz*16,b['z0']);az1=min(cz*16+16,b['z1'])
            if ax1>ax0 and az1>az0:
                mask[az0-cz*16:az1-cz*16,ax0-cx*16:ax1-cx*16]=self.labels[az0-b['z0']:az1-b['z0'],ax0-b['x0']:ax1-b['x0']]==item['index']
            a*=mask[None];a[:max(0,q['y0']-1)]=0
        return a,biomes

    def mesh(self,cx,cz,cut=255,building=None,clip_bounds=None):
        a,biomes=self.clipped(cx,cz,cut,building,clip_bounds);b=self.report['bounds']
        padded=np.pad(a,((1,1),(1,1),(1,1)))
        for dx,dz in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor,_=self.clipped(cx+dx,cz+dz,cut,building,clip_bounds)
            if dx==-1:padded[1:-1,1:-1,0]=neighbor[:,:,-1]
            if dx==1:padded[1:-1,1:-1,-1]=neighbor[:,:,0]
            if dz==-1:padded[1:-1,0,1:-1]=neighbor[:,-1,:]
            if dz==1:padded[1:-1,-1,1:-1]=neighbor[:,0,:]
        neighbours=[padded[1:-1,1:-1,:-2],padded[1:-1,1:-1,2:],padded[:-2,1:-1,1:-1],padded[2:,1:-1,1:-1],padded[1:-1,:-2,1:-1],padded[1:-1,2:,1:-1]]
        opaque=[];transparent=[];stats=Counter();special=[]
        # Batch full cubes by state and direction, keeping draw calls per chunk at two.
        for face,n in enumerate(neighbours):
            exposed=(a!=0)&~self.solid[n]&~((n==a)&np.isin(a,[c for c,s in self.records.items() if s['visual'].get('geometry') in {'fluid','alpha_cube'}]))
            for code in np.unique(a[exposed]):
                if int(code) not in self.records: continue
                record=self.records[int(code)];g=record['visual'].get('geometry','cube')
                if g not in {'cube','alpha_cube'}: continue
                y,z,x=np.nonzero(exposed&(a==code));count=len(x)
                if not count: continue
                positions=np.stack((x,y,z),axis=1).astype(np.float32)
                verts=positions[:,None,:]+FACES[face][None]
                verts[:,:,0]+=cx*16-b['x0'];verts[:,:,2]+=cz*16-b['z0']
                uv=np.broadcast_to(self.uv(self.sprites[int(code)][face]),(count,4,2))
                biome_values,inverse=np.unique(biomes[z,x],return_inverse=True)
                colors=np.asarray([self.tint(int(code),v,face) for v in biome_values],np.float32)[inverse]*LIGHT[face]
                color=np.broadcast_to(colors[:,None,:],(count,4,3))
                target=transparent if (int(code)>>4) in {20,79} else opaque
                target.append(np.concatenate((verts,uv,color),axis=2)[:,TRI,:].reshape(-1,8));stats[g]+=count
        for code in np.unique(a):
            if not code or int(code) not in self.records: continue
            record=self.records[int(code)];v=record['visual'];g=v.get('geometry','cube');bid,meta=divmod(int(code),16)
            if g in {'cube','alpha_cube'}: continue
            candidate=(a==code)
            if g=='fluid':
                same_family=lambda n: np.isin(n>>4,[8,9] if bid in {8,9} else [10,11])
                visible=np.zeros_like(candidate)
                for n in neighbours: visible|=~self.solid[n]&~same_family(n)
                candidate &= visible
            # Batch repeated plants and partial shapes instead of allocating per block.
            candidate &= ~np.logical_and.reduce([self.solid[n] for n in neighbours])
            y,z,x=np.nonzero(candidate)
            if not len(x):continue
            link_values=np.zeros(len(x),np.uint8)
            if g=='connected':
                for bit,n in enumerate([0,1,4,5]):link_values|=(neighbours[n][y,z,x]!=0).astype(np.uint8)<<bit
            for link_value in np.unique(link_values):
                group=link_values==link_value;gy,gz,gx=y[group],z[group],x[group]
                if g=='cross':
                    quads=[np.array([[.08,0,.08],[.92,0,.92],[.92,1,.92],[.08,1,.08]]),np.array([[.92,0,.08],[.08,0,.92],[.08,1,.92],[.92,1,.08]])]
                    faceids=[3,3]
                    for quad in quads:quad[:,1]*=float(v.get('height',1))
                else:
                    links=[bool(int(link_value)&(1<<bit)) for bit in range(4)]
                    quads=[];faceids=[]
                    for box in boxes(v,bid,meta,links):
                        lo=np.array(box[:3]);hi=np.array(box[3:])
                        for face in range(6):quads.append(FACES[face]*(hi-lo)+lo);faceids.append(face)
                for quad,face in zip(quads,faceids):
                    keep=np.ones(len(gx),bool)
                    if g=='fluid':
                        nc=neighbours[face][gy,gz,gx];family=[8,9] if bid in {8,9} else [10,11]
                        keep=~np.isin(nc>>4,family)&~self.solid[nc]
                    py,pz,px=gy[keep],gz[keep],gx[keep];count=len(px)
                    if not count:continue
                    origin=np.stack((px+cx*16-b['x0'],py,pz+cz*16-b['z0']),axis=1).astype(np.float32)
                    vert=origin[:,None,:]+quad[None]
                    uv=np.broadcast_to(self.uv(self.sprites[int(code)][face]),(count,4,2))
                    biome_values,inverse=np.unique(biomes[pz,px],return_inverse=True)
                    colors=np.asarray([self.tint(int(code),bv,face) for bv in biome_values],np.float32)[inverse]*LIGHT[face]
                    color=np.broadcast_to(colors[:,None,:],(count,4,3))
                    target=transparent if g=='fluid' else opaque
                    target.append(np.concatenate((vert,uv,color),axis=2)[:,TRI,:].reshape(-1,8).astype(np.float32));stats[g]+=count
        arrays=[np.concatenate(parts).astype('<f4') if parts else np.zeros((0,8),'<f4') for parts in [opaque,transparent]]
        header=json.dumps({'vertices':[len(ar) for ar in arrays],'stride':8,'chunk':[cx,cz],'cut':cut,'building':building,'bounds':clip_bounds,'faces':dict(stats)},separators=(',',':')).encode()
        header+=b' '*((-len(header))%4)
        return np.array([len(header)],'<u4').tobytes()+header+b''.join(ar.tobytes() for ar in arrays)
