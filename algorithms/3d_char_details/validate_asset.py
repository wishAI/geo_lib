"""Validate the actual exported GLB with only Python's standard library."""
from pathlib import Path
import hashlib
import json
import math
import struct

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/landau_v10'

def validate():
    data=(OUT/'landau_character.glb').read_bytes()
    magic,version,length=struct.unpack_from('<4sII',data)
    assert magic==b'glTF' and version==2 and length==len(data), 'Invalid GLB header'
    chunks={};offset=12
    while offset<len(data):
        size,kind=struct.unpack_from('<II',data,offset);offset+=8
        chunks[kind]=data[offset:offset+size];offset+=size
    gltf=json.loads(chunks[0x4E4F534A]);binary=chunks[0x004E4942]
    report=json.loads((OUT/'asset_report.json').read_text())
    assert hashlib.sha256(data).hexdigest()==report['glb_sha256'], 'Report is for a different GLB'
    assert not any('uri' in x for x in gltf.get('buffers',[])+gltf.get('images',[])), 'External asset dependency'
    body_report=report.get('body_reconstruction')
    if body_report:
        assert report['version']==body_report['revision'] and body_report['revision'] in [3,4], 'Unexpected body revision'
        preservation=report['neutral_preservation']
        assert preservation['revision']==body_report['revision'] and preservation['face_exactly_preserved'], 'Missing facial preservation contract'
        before=body_report['face_hashes_before'];after=body_report['face_hashes_after']
        assert before and before==after and body_report['face_exactly_preserved'], 'Protected facial data changed'
        assert {'Head','Lash_L','Lash_R','UpperLid_L','UpperLid_R'}<=before.keys(), 'Incomplete facial snapshot'
        assert all(len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in before.values()), 'Invalid facial digest'
        assert math.isfinite(body_report['head_rigid_lift']) and preservation['head_rigid_lift']==body_report['head_rigid_lift']
    else:
        assert report['neutral_preservation']['max_position_error']==0, 'Neutral geometry changed'
        assert report['neutral_preservation']['retained_faces']==report['neutral_preservation']['source_faces']==50000
    assert not report['validation']['errors'] and report['validation']['invalid_skin_vertices']==0
    components={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
    formats={5126:('f',4),5125:('I',4),5123:('H',2),5121:('B',1),5122:('h',2),5120:('b',1)}
    def read_rows(view_index, offset, count, width, component):
        view=gltf['bufferViews'][view_index];fmt,size=formats[component]
        stride=view.get('byteStride',width*size);start=view.get('byteOffset',0)+offset
        assert start+max(0,count-1)*stride+width*size<=len(binary)
        return [struct.unpack_from('<'+fmt*width,binary,start+i*stride) for i in range(count)]
    def values(index):
        a=gltf['accessors'][index];width=components[a['type']]
        rows=read_rows(a['bufferView'],a.get('byteOffset',0),a['count'],width,a['componentType']) if 'bufferView' in a else [(0,)*width]*a['count']
        sparse=a.get('sparse')
        if sparse:
            ids=sparse['indices'];v=sparse['values']
            indices=read_rows(ids['bufferView'],ids.get('byteOffset',0),sparse['count'],1,ids['componentType'])
            replacements=read_rows(v['bufferView'],v.get('byteOffset',0),sparse['count'],width,a['componentType'])
            for (i,),row in zip(indices,replacements):
                assert 0<=i<len(rows);rows[i]=row
        return rows
    triangles=0;controls=set();skinned=0
    for mesh in gltf['meshes']:
        assert all(w==0 for w in mesh.get('weights',[])), 'Non-neutral default expression'
        names=mesh.get('extras',{}).get('targetNames',[]);controls.update(names)
        name=mesh.get('name','')
        assert not name.startswith('LowerLid'), 'Unexpected lower eyelid'
        if name.startswith(('EyeShell_','Iris_','RoundIris_','Pupil_','Catchlight')):
            assert not any(n.startswith(('eyeBlink','eyeSquint')) for n in names), name+' changes shape during blink'
        if name.startswith('Lash_'):
            assert 'eyeBlink'+name[-1] in names, 'Original lash has no blink'
            count=sum(gltf['accessors'][p['indices']]['count']//3 for p in mesh['primitives'])
            assert count==report['separated_parts'][name], 'Original lash topology was replaced'
        for primitive in mesh['primitives']:
            attrs=primitive['attributes'];assert 'JOINTS_0' in attrs and 'WEIGHTS_0' in attrs
            assert primitive.get('mode',4)==4
            triangles+=gltf['accessors'][primitive['indices']]['count']//3
            for row in values(attrs['WEIGHTS_0']):
                assert abs(sum(row)-1)<1e-4 and all(w>=0 for w in row), 'Bad skin weights'
                skinned+=1
            for index in [attrs['POSITION']]+[t['POSITION'] for t in primitive.get('targets',[])]:
                assert all(math.isfinite(x) for row in values(index) for x in row), 'Non-finite deformation'
            assert len(names)==len(primitive.get('targets',[]))
    bone_counts=[len(s['joints']) for s in gltf['skins']]
    assert max(bone_counts)==71
    assert {'eyeBlinkL','eyeBlinkR','mouthSmile','jawDrop','eyeLookUpL','eyeSize','cheekFullness'}<=controls
    assert triangles==report['validation']['triangles'], 'Unexpected extra or missing exported geometry'
    assert len(gltf['meshes'])==report['validation']['mesh_count'], 'Unexpected inspection meshes in export'
    assert len(gltf['scenes'])==1, 'Export must contain only the active character scene'
    hidden=[n['name'] for n in gltf['nodes'] if n.get('extras',{}).get('default_hidden')]
    if body_report:
        nodes={n['name']:n for n in gltf['nodes'] if 'name' in n}
        body_name=body_report['body_mesh'];assert body_name=='Body_Complete' and body_name in nodes, 'Complete body missing'
        assert 'Body_UnderClothes' not in nodes, 'Obsolete body approximation remains'
        body=nodes[body_name];extras=body.get('extras',{})
        assert 'mesh' in body and 'skin' in body, 'Complete body is not independently skinned'
        assert extras.get('complete_under_outfit') and not extras.get('default_hidden'), 'Complete body is hidden or incomplete'
        assert extras.get('part_type')=='inferred_body', 'Body visibility metadata missing'
        joints=gltf['skins'][body['skin']]['joints']
        assert len(joints)==71
        expected_garments={'Vest','Sleeve_L','Sleeve_R','Cuff_L','Cuff_R','Trousers','Boot_L','Boot_R'}
        assert set(body_report['garments'])==expected_garments, 'Incomplete garment inventory'
        mesh_ids={body['mesh']}
        for name in body_report['garments']:
            assert name in nodes, 'Missing garment: '+name
            garment=nodes[name]
            assert garment.get('extras',{}).get('part_type')=='clothing', 'Garment visibility metadata missing: '+name
            assert 'mesh' in garment and 'skin' in garment, 'Garment not independently skinned: '+name
            assert garment['mesh'] not in mesh_ids, 'Body/garment mesh data is shared: '+name
            mesh_ids.add(garment['mesh'])
            assert gltf['skins'][garment['skin']]['joints']==joints, 'Garment skeleton differs: '+name
        assert before.keys()<=nodes.keys(), 'Protected facial objects missing from export'
    else:
        assert hidden==['Body_UnderClothes']
    result={'status':'passed','triangles':triangles,
        'skinned_vertices':skinned,'bones':max(bone_counts),'controls':len(controls),
        'source_neutral_positions':'face local data exact; whole head rigidly relocated' if body_report else 'exact',
        'original_lashes_animated':True,'ocular_blink_deformation':False,'lower_eyelid':False,'embedded_images':len(gltf.get('images',[])),
        'scope':'Data integrity. Likeness, extreme poses and production lip sync are separate art gates.'}
    if body_report:
        result.update({'body_revision':body_report['revision'],'complete_body_visible':True,'garments_share_body_skeleton':True,
            'protected_facial_objects':len(before),'head_rigid_lift':body_report['head_rigid_lift']})
    else:
        result.update({'triangles_including_hidden_body':triangles,'visible_source_triangles':50000})
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':validate()
