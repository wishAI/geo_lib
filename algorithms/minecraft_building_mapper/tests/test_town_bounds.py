import tempfile
import unittest
from pathlib import Path
from algorithms.minecraft_building_mapper.town_bounds import BoundsStore, validate_bounds

class BoundsTests(unittest.TestCase):
    town={'x0':-100,'x1':100,'z0':-100,'z1':100}
    box={'x0':-10,'y0':64,'z0':-10,'x1':10,'y1':255,'z1':10}
    def test_all_six_faces_and_invalid_coordinates(self):
        self.assertEqual(validate_bounds(self.box,self.town),self.box)
        for change in [{'y1':257},{'y0':255},{'x0':-101},{'z1':101},{'x1':-10},{'y1':254.5},{'y0':True}]:
            with self.subTest(change=change),self.assertRaises(ValueError):validate_bounds(self.box|change,self.town)
    def test_overrides_persist_and_restore_without_touching_source(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'town.json';source.write_text('source unchanged')
            report={'bounds':self.town,'buildings':[{'id':'building-0001'}]}
            store=BoundsStore(d,report);store.save({'id':'building-0001','bounds':self.box})
            self.assertEqual(BoundsStore(d,report).read()['buildings']['building-0001'],self.box)
            with self.assertRaises(ValueError):store.save({'id':'../world','bounds':self.box})
            self.assertEqual(store.save({'id':'building-0001','reset':True})['buildings'],{})
            self.assertEqual(source.read_text(),'source unchanged')
