"""Source-surface segmentation using geometric creases, never texture labels.

A small landmark seed identifies the requested component; graph cut chooses its
boundary on the mesh. A fused or featureless region can fail this diagnostic.
"""
import math
import json
from collections import deque
from pathlib import Path
import numpy as np
import bpy

ROOT=Path(__file__).resolve().parent

class Cut:
    def __init__(self,n):self.graph=[[] for _ in range(n)]
    def add(self,a,b,capacity):
        self.graph[a].append([b,float(capacity),len(self.graph[b])])
        self.graph[b].append([a,0.,len(self.graph[a])-1])
    def solve(self,source,sink):
        while True:
            level=[-1]*len(self.graph);level[source]=0;q=deque([source])
            while q:
                a=q.popleft()
                for b,c,r in self.graph[a]:
                    if c>1e-12 and level[b]<0:level[b]=level[a]+1;q.append(b)
            if level[sink]<0:break
            cursor=[0]*len(self.graph)
            def push(a,flow):
                if a==sink:return flow
                while cursor[a]<len(self.graph[a]):
                    e=self.graph[a][cursor[a]];b,c,r=e
                    if c>1e-12 and level[b]==level[a]+1:
                        sent=push(b,min(flow,c))
                        if sent>1e-12:e[1]-=sent;self.graph[b][r][1]+=sent;return sent
                    cursor[a]+=1
                return 0.
            while push(source,1e10)>1e-12:pass
        seen={source};q=deque([source])
        while q:
            a=q.popleft()
            for b,c,r in self.graph[a]:
                if c>1e-12 and b not in seen:seen.add(b);q.append(b)
        return seen

def iris_cut(obj,sign=1,crease_scale=.12):
    mesh=obj.data;verts=np.array([v.co[:] for v in mesh.vertices]);faces=np.array([p.vertices[:] for p in mesh.polygons])
    centers=verts[faces].mean(1);x,y,z=centers.T
    ns=np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]])
    ns/=np.maximum(np.linalg.norm(ns,axis=1)[:,None],1e-12)
    cx=sign*.0615;cz=.688
    radius=np.sqrt(((x-cx)/.0235)**2+((z-cz)/.026)**2)
    domain=(radius<1.65)&(y<-.069)&(ns[:,1]<.5)
    ids=np.where(domain)[0];local={int(v):i for i,v in enumerate(ids)}
    welded={};wi=[]
    for v in verts:
        key=tuple(np.round(v,6));wi.append(welded.setdefault(key,len(welded)))
    edge_map={};links=[]
    for fi in ids:
        face=faces[fi]
        for i,j in zip(face,np.roll(face,-1)):
            edge=tuple(sorted((wi[i],wi[j])))
            if edge in edge_map:
                other=edge_map[edge]
                theta=math.acos(float(np.clip(np.dot(ns[fi],ns[other]),-1,1)))
                length=float(np.linalg.norm(verts[i]-verts[j]))
                links.append((int(fi),int(other),theta,length,tuple(int(a) for a in (i,j))))
            else:edge_map[edge]=int(fi)
    n=len(ids);flow=Cut(n+2);source=n;sink=n+1
    for fi in ids:
        if radius[fi]<.28:flow.add(source,local[int(fi)],1000)
        if radius[fi]>1.35:flow.add(local[int(fi)],sink,1000)
    for a,b,theta,length,edge in links:
        cost=length/(1+(theta/crease_scale)**4)+.000005
        flow.add(local[a],local[b],cost);flow.add(local[b],local[a],cost)
    reachable=flow.solve(source,sink);selected={int(fi) for fi in ids if local[int(fi)] in reachable}
    boundary=[(a,b,t,l,e) for a,b,t,l,e in links if (a in selected)!=(b in selected)]
    pts=verts[faces[list(selected)].ravel()]
    report={'component':'iris_surface_'+('L' if sign>0 else 'R'),'faces':len(selected),'domain_faces':n,
        'boundary_edges':len(boundary),'mean_boundary_bend_degrees':float(np.mean([t*180/math.pi for a,b,t,l,e in boundary])),
        'bounds':[pts.min(0).tolist(),pts.max(0).tolist()],
        'method':'Geometric dihedral-weighted minimum cut between interior and exterior landmarks; no UV/color input.',
        'crease_scale':crease_scale}
    return selected,report

def diagnose():
    obj=bpy.data.objects.get('material')
    if obj is None:raise RuntimeError('Load the untouched source_import.blend first')
    clay=bpy.data.materials.new('Source clay');clay.diffuse_color=(.55,.55,.55,1);clay.use_nodes=True
    clay.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.55,.55,.55,1)
    clay.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.9
    selected=bpy.data.materials.new('Geometry selection');selected.diffuse_color=(.03,.32,.5,1);selected.use_nodes=True
    selected.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=(.03,.32,.5,1)
    obj.data.materials.clear();obj.data.materials.append(clay);obj.data.materials.append(selected)
    ids,report=iris_cut(obj)
    for face in obj.data.polygons:face.material_index=1 if face.index in ids else 0
    (ROOT/'outputs/landau_v10/geometric_iris_selection.json').write_text(json.dumps(report,indent=2))
    (ROOT/'outputs/landau_v10/geometric_iris_faces.json').write_text(json.dumps(sorted(ids)))
    print(json.dumps(report))

if __name__=='__main__':diagnose()

def component_regions(obj):
    """Find the source's actual sculpted facial regions through dihedral barriers."""
    verts=np.array([v.co[:] for v in obj.data.vertices]);faces=np.array([p.vertices[:] for p in obj.data.polygons])
    unique,inv=np.unique(np.round(verts,6),axis=0,return_inverse=True);wfaces=inv[faces]
    centers=unique[wfaces].mean(1)
    normals=np.cross(unique[wfaces[:,1]]-unique[wfaces[:,0]],unique[wfaces[:,2]]-unique[wfaces[:,0]])
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
    edges={};pairs=[];adj=[set() for _ in faces]
    for i,ids in enumerate(wfaces):
        for a,b in zip(ids,np.roll(ids,-1)):
            edge=tuple(sorted((int(a),int(b))))
            if edge in edges:
                other=edges[edge];pairs.append((other,i));adj[other].add(i);adj[i].add(other)
            else:edges[edge]=i
    pairs=np.array(pairs);angles=np.arccos(np.clip(np.sum(normals[pairs[:,0]]*normals[pairs[:,1]],axis=1),-1,1))*180/math.pi
    roots={}
    for limit in [15,35]:
        parent=list(range(len(faces)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for a,b in pairs[angles<limit]:parent[find(a)]=find(b)
        roots[limit]=np.array([find(i) for i in range(len(faces))])
    seeds={'Face_Cream':((.106,-.075,.751),15),'Nose':((0,-.152,.650),35)}
    for sign,side in [(1,'L'),(-1,'R')]:
        seeds.update({
            'EyeShell_'+side:((sign*.064,-.103,.689),35),
            'Lash_'+side:((sign*.095,-.099,.722),35),
            'Brow_'+side:((sign*.077,-.109,.757),35),
            'InnerEar_'+side:((sign*.103,.050,1.012),35)})
    result={};reports={}
    for name,(seed,limit) in seeds.items():
        idx=int(np.argmin(np.linalg.norm(centers-seed,axis=1)))
        region=set(np.flatnonzero(roots[limit]==roots[limit][idx]).tolist())
        if len(region)>5000:raise RuntimeError(name+' leaked outside its geometric component')
        # Fill only enclosed one- or two-face holes, never eye/brow sockets.
        candidates={j for i in region for j in adj[i] if j not in region}
        for candidate in sorted(candidates):
            if candidate in region:continue
            hole={candidate};stack=[candidate]
            while stack and len(hole)<=2:
                i=stack.pop()
                for j in adj[i]:
                    if j not in region and j not in hole:hole.add(j);stack.append(j)
            if len(hole)<=2 and all(len(adj[i])==3 for i in hole):region.update(hole)
        result[name]=region
        mask=np.zeros(len(faces),bool);mask[list(region)]=True
        crossing=mask[pairs[:,0]]!=mask[pairs[:,1]]
        reports[name]={'faces':len(region),'seed':seed,'barrier_degrees':limit,'boundary_edges':int(sum(crossing)),
            'mean_boundary_bend_degrees':float(np.mean(angles[crossing]))}
    removed=set().union(*result.values());remaining=set(range(len(faces)))-removed
    additions=[]
    while remaining:
        seed=remaining.pop();region={seed};stack=[seed]
        while stack:
            i=stack.pop()
            for j in adj[i]:
                if j in remaining:remaining.remove(j);region.add(j);stack.append(j)
        if len(region)>60:continue
        border={j for i in region for j in adj[i] if j not in region}
        cream_neighbors=border & result['Face_Cream']
        if border and len(cream_neighbors)/len(border)>=.6:
            result['Face_Cream'].update(region);additions.extend(region)
    reports['cream_residual_repair']={'faces':len(additions),'source_face_ids':sorted(additions),'method':'Enclosed residual components with at least 60% cream adjacency.'}
    for sign,side in [(1,'L'),(-1,'R')]:
        region,info=iris_cut(obj,sign)
        result['Iris_'+side]=region;reports['Iris_'+side]=info
    # Explicit precedence resolves shared rim triangles; every source face gets
    # exactly one owner when the main builder partitions the full source mesh.
    priority=['Face_Cream','EyeShell_L','EyeShell_R','InnerEar_L','InnerEar_R','Lash_L','Lash_R','Brow_L','Brow_R','Nose','Iris_L','Iris_R']
    owner={}
    for name in priority:
        for i in result[name]:owner[i]=name
    return owner,reports
