"""Synthetic regression fixtures; never load or write real save data."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from algorithms.minecraft_building_mapper.town import unpack_chunk,recognize
from algorithms.minecraft_building_mapper.town_mesh import boxes,FACES,Mesher
import json
import struct
from algorithms.minecraft_building_mapper.world import ChunkView
from algorithms.minecraft_building_mapper.nbt import loads
from algorithms.minecraft_building_mapper.tests.helpers import legacy_chunk

class TownTests(unittest.TestCase):
    def test_high_legacy_ids_and_metadata_roundtrip(self):
        # Cover both low and high nibbles and the Add array's 12-bit IDs.
        class Chunk:
            modern={}
            legacy={0:(bytes([252,253]+[0]*4094),bytes([0xA9]+[0]*2047),bytes([0xFF]+[0]*2047))}
        dense=unpack_chunk(Chunk())
        self.assertEqual(int(dense[0,0,0]),(4092<<4)|9)
        self.assertEqual(int(dense[0,0,1]),(4093<<4)|10)
        self.assertEqual(np.count_nonzero(dense),2)

    def test_road_separates_two_buildings_not_flat_paving(self):
        h=np.full((60,60),64,np.int16);g=h.copy();rh=h.copy();bc=np.zeros_like(h);rc=np.full_like(h,4<<4);valid=np.ones_like(h,bool)
        for sl in [(slice(10,25),slice(10,23)),(slice(10,25),slice(28,42))]:
            rh[sl]=74;h[sl]=74;bc[sl]=8;rc[sl]=5<<4
        # Shared ground paving should never glue neighbouring buildings.
        bc[25:40,5:50]=1
        b={'x0':-100,'z0':200,'x1':-40,'z1':260}
        result,labels,roads,evidence=recognize(h,g,bc,rh,rc,valid,b)
        self.assertEqual(len(result),2)
        self.assertTrue(roads[15,25]);self.assertFalse(labels[35,15])
        self.assertEqual(result[0]['bounds']['x0'],-90)

    def test_trees_and_unallocated_columns_do_not_become_buildings(self):
        h=np.full((30,30),80,np.int16);g=np.full_like(h,63);bc=np.zeros_like(h);valid=np.ones_like(h,bool)
        result,*_=recognize(h,g,bc,g,np.zeros_like(h),valid,{'x0':0,'z0':0})
        self.assertEqual(result,[])
        bc[:]=20;valid[:]=False
        result,*_=recognize(h,g,bc,h,np.full_like(h,5<<4),valid,{'x0':0,'z0':0})
        self.assertEqual(result,[])

    def test_slab_stair_and_connected_geometry(self):
        self.assertEqual(boxes({'geometry':'slab'},44,8)[0],(0,.5,0,1,1,1))
        stair=boxes({'geometry':'stair'},53,0);self.assertEqual(len(stair),2)
        self.assertEqual(stair[1],(.5,.5,0,1,1,1))
        self.assertEqual(len(boxes({'geometry':'connected'},85,0,(True,True,False,False))),5)
        expected=[[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]
        for face,normal in zip(FACES,expected):
            np.testing.assert_array_equal(np.cross(face[1]-face[0],face[2]-face[0]),normal)

class MeshIntegrationTests(unittest.TestCase):
    def mesher(self,blocks,neighbour=None):
        m=Mesher.__new__(Mesher);m.report={'bounds':{'x0':0,'x1':32,'z0':0,'z1':16}}
        m.records={16:{'key':'legacy:1:0','visual':{'geometry':'cube'}},144:{'key':'legacy:9:0','visual':{'geometry':'fluid'}},592:{'key':'legacy:37:0','visual':{'geometry':'cross','height':.8}}}
        m.catalog=type('Catalog',(),{'block_colorizer':lambda self,key:None})();m.solid=np.zeros(65536,bool);m.solid[16]=True
        m.sprites={c:[0]*6 for c in m.records};m.atlas_cells=1;m.atlas_size=34
        def load(x,z):
            a=blocks if (x,z)==(0,0) else neighbour if (x,z)==(1,0) and neighbour is not None else np.zeros_like(blocks)
            return a.copy(),np.ones((16,16),np.uint8)
        m.load=load;return m
    def decode(self,raw):
        n=struct.unpack('<I',raw[:4])[0];head=json.loads(raw[4:4+n]);v=np.frombuffer(raw[4+n:],dtype='<f4').reshape(-1,8)
        self.assertTrue(np.isfinite(v).all());self.assertEqual(len(v),sum(head['vertices']));return head,v
    def test_crossed_flower_survives_with_its_full_height(self):
        a=np.zeros((256,16,16),np.uint16);a[64,8,8]=592
        head,v=self.decode(self.mesher(a).mesh(0,0));self.assertEqual(head['faces']['cross'],2)
        self.assertEqual(len(v),12);self.assertAlmostEqual(float(v[:,1].max()),64.8,places=4)
    def test_water_volume_emits_only_its_exposed_boundary(self):
        a=np.zeros((256,16,16),np.uint16);a[60:64,4:8,4:8]=144
        head,v=self.decode(self.mesher(a).mesh(0,0));self.assertEqual(head['faces']['fluid'],96)
    def test_isolated_building_excludes_adjacent_structure(self):
        a=np.zeros((256,16,16),np.uint16);a[64,8,8]=16;a[64,8,10]=16;a[20,8,8]=16
        m=self.mesher(a);m.labels=np.zeros((16,32),np.uint16);m.labels[8,8]=1;m.labels[8,10]=2
        m.buildings={'building-0001':{'index':1,'bounds':{'x0':8,'z0':8,'x1':9,'z1':9,'y0':64,'y1':65}}}
        head,v=self.decode(m.mesh(0,0,building='building-0001'))
        self.assertEqual(head['building'],'building-0001');self.assertEqual(head['faces']['cube'],6)
        self.assertEqual(float(v[:,0].max()),9);self.assertEqual(float(v[:,1].min()),64)

    def test_expanded_box_includes_new_columns_and_clips_all_sides(self):
        a=np.zeros((256,16,16),np.uint16)
        a[64,8,8]=16;a[64,8,10]=16;a[63,8,8]=16;a[66,8,8]=16;a[64,7,8]=16
        m=self.mesher(a)
        q={'x0':8,'y0':64,'z0':8,'x1':11,'y1':66,'z1':9}
        head,v=self.decode(m.mesh(0,0,building='building-0001',clip_bounds=q))
        self.assertEqual(head['faces']['cube'],12)
        self.assertEqual(tuple(v[:,:3].min(axis=0)),(8,64,8))
        self.assertEqual(tuple(v[:,:3].max(axis=0)),(11,65,9))

    def test_cross_chunk_culling_and_cutoff(self):
        a=np.zeros((256,16,16),np.uint16);a[64,8,15]=16;n=np.zeros_like(a);n[64,8,0]=16
        m=self.mesher(a,n);head,v=self.decode(m.mesh(0,0));self.assertEqual(head['faces']['cube'],5)
        head,v=self.decode(m.mesh(0,0,63));self.assertEqual(len(v),0)
        m.report['bounds']['x1']=16;head,v=self.decode(m.mesh(0,0));self.assertEqual(head['faces']['cube'],6)

if __name__=='__main__':unittest.main()
