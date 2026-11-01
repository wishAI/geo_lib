import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "acot_cn_builder.py"
SPEC = importlib.util.spec_from_file_location("acot_cn_builder", MODULE_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class AcotChineseBuilderTests(unittest.TestCase):
    def test_markup_round_trip(self):
        source = "§YShield§! protects $COUNTRY$ on [Root.GetName].\\nCost: £energy£10."
        protected, token_map = builder.protect_text(source)
        self.assertNotIn("$COUNTRY$", protected)
        self.assertNotIn("[Root.GetName]", protected)
        self.assertEqual(builder.restore_text(protected, token_map), source)

    def test_glossary_matches_existing_acot_terms(self):
        protected, token_map = builder.protect_text(
            "The Stellarborne guard the Outer Gate with Gatekeepers and the Light."
        )
        restored = builder.restore_text(protected, token_map)
        self.assertIn("星裔", restored)
        self.assertIn("外星门", restored)
        self.assertIn("守门人", restored)
        self.assertIn("圣光", restored)

    def test_parse_and_validate_preserve_keys_and_markup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "english"
            output = root / "simp_chinese"
            source.mkdir()
            (source / "sample_l_english.yml").write_text(
                '\ufeffl_english:\n sample:0 "Hello $NAME$ [Root.GetName]§!" # note\n',
                encoding="utf-8",
            )
            tree = builder.parse_tree(source)
            builder.write_localisation(
                tree,
                {"sample": "你好 $NAME$ [Root.GetName]§!"},
                output,
            )
            result = builder.validate_pair(tree, output)
            self.assertEqual(result["sourceKeys"], 1)
            self.assertEqual(result["translatedKeys"], 1)
            self.assertEqual(result["markupErrors"], 0)


if __name__ == "__main__":
    unittest.main()
