import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
import unittest

Store=importlib.import_module('algorithms.3d_char_details.preset_store').PresetStore

class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.store=Store(self.tmp.name)
        self.settings={'version':1,'assetHash':'a'*64,'bodyFrame':{'scale':1.1,'offset':.012},'outfit':{'Boot_L':{'bootWidth':.4}},'links':{'boots':False}}
    def save(self,n):return self.store.update({'action':'save','name':str(n),'settings':self.settings})
    def test_concurrent_saves_survive_reload_and_metadata_edits(self):
        with ThreadPoolExecutor(max_workers=8) as pool: records=list(pool.map(self.save,range(24)))
        self.assertEqual(len({r['id'] for r in records}),24)
        reloaded=Store(self.tmp.name);self.assertEqual(len(reloaded.list()['entries']),24)
        r=records[0]
        for req in [{'action':'rename','name':'Fit A'},{'action':'archive','archived':True},{'action':'archive','archived':False}]:
            reloaded.update({**req,'id':r['id']})
            self.assertEqual(reloaded.get(r['id'])['settings'],self.settings)
            self.assertEqual(reloaded.get(r['id'])['createdAt'],r['createdAt'])
        self.assertEqual(reloaded.get(r['id'])['name'],'Fit A')
        self.assertFalse(reloaded.get(r['id'])['archived'])
    def test_bad_input_does_not_change_history(self):
        self.save('Good');before=self.store.path.read_bytes()
        for req in [{'action':'save','name':' ','settings':self.settings},{'action':'save','name':'Bad','settings':{**self.settings,'bodyFrame':{'scale':float('nan')}}},{'action':'archive','id':'../../outside','archived':True},{'action':'rename','id':'missing','name':'Bad'},{'action':'save','name':'Bad','settings':{'version':1,'assetHash':'wrong'}}]:
            with self.assertRaises((ValueError,KeyError)):self.store.update(req)
            self.assertEqual(before,self.store.path.read_bytes())

if __name__=='__main__':unittest.main()
