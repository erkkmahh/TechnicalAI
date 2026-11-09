"""Synthetic offline coverage for optional Hybrid parameter sources."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import hybrid_search as hybrid


class ParameterSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        (self.folder / "synthetic.pdf").touch()

    def fixtures(self, number=4567, bit=2, symbol="XYZ", page=10):
        text = f"{number} #{bit}\n{symbol}\nFictional description\n0: Fictional zero\n1: Fictional one"
        rows = [{"text":line, "page":page if i < 2 else page+1} for i,line in enumerate(text.splitlines())]
        metadata = {"mode":"parameter_range", "fallback_reason":None, "target_page":page,"source_layout":rows}
        results = [{"text":f"{number} #{bit}\n{symbol}\n", "filename":"synthetic.pdf", "page":page,"rrf_score":0.03}]
        return text, metadata, results

    def run_case(self, query="4567#2 XYZ", **changes):
        text, metadata, results = self.fixtures()
        metadata.update(changes)
        before = copy.deepcopy(results)
        with patch.object(hybrid.parameter_parser,"extract_source",return_value=(metadata,text,text)) as extract, patch.object(hybrid.parameter_parser,"extract_main_description",return_value={"main_description_status":"VERIFIED","main_description":"Fictional description","source_line_indices":[2]}):
            actual = hybrid.expand_parameter_sources(query,results,self.folder)
        return before,actual,extract

    def test_cross_page_and_original_fields(self):
        before,actual,extract = self.run_case()
        self.assertEqual({k:v for k,v in actual[0].items() if k!="parameter_source"},before[0])
        self.assertEqual(actual[0]["parameter_source"]["pages"],[10,11])
        self.assertEqual(actual[0]["parameter_source"]["description_pages"],[11])
        self.assertEqual(extract.call_args.kwargs["pdf_path"],self.folder.resolve()/"synthetic.pdf")

    def test_wrong_symbol_and_missing_parameter(self):
        for query in ["4567#2 WRONG","9999#1 XYZ"]:
            before,actual,extract=self.run_case(query)
            self.assertEqual(actual,before)
            extract.assert_not_called()

    def test_only_explicit_single_identity(self):
        for query in ["Explain 4567#2 XYZ","4567#2","4567#2 XYZ 4568#1 ABC"]:
            before,actual,extract=self.run_case(query)
            self.assertEqual(actual,before)
            extract.assert_not_called()

    def test_fallback_and_missing_layout_rejected(self):
        for changes in [{"mode":"three_page_context"},{"fallback_reason":"ambiguous"},{"source_layout":None}]:
            before,actual,_=self.run_case(**changes)
            self.assertEqual(actual,before)

    def test_parser_failure_preserves_results(self):
        _,_,results=self.fixtures()
        before=copy.deepcopy(results)
        with patch.object(hybrid.parameter_parser,"extract_source",side_effect=ValueError("ambiguous")):
            self.assertEqual(hybrid.expand_parameter_sources("4567#2 XYZ",results,self.folder),before)

    def test_next_heading_and_unverified_description_rejected(self):
        text,metadata,results=self.fixtures()
        for extra,status in [("\n4568 #1\nABC", "VERIFIED"),("","NOT_VERIFIED")]:
            source=text+extra
            metadata["source_layout"]=[{"text":line,"page":10} for line in source.splitlines()]
            with patch.object(hybrid.parameter_parser,"extract_source",return_value=(metadata,source,source)), patch.object(hybrid.parameter_parser,"extract_main_description",return_value={"main_description_status":status}):
                actual=hybrid.expand_parameter_sources("4567#2 XYZ",copy.deepcopy(results),self.folder)
                self.assertNotIn("parameter_source",actual[0])

    def test_same_page_identities_preserve_ranking(self):
        for number,bit,symbol in [(5678,1,"ABC"),(12345,6,"DEF")]:
            text,metadata,results=self.fixtures(number,bit,symbol)
            for row in metadata["source_layout"]:row["page"]=10
            with patch.object(hybrid.parameter_parser,"extract_source",return_value=(metadata,text,text)), patch.object(hybrid.parameter_parser,"extract_main_description",return_value={"main_description_status":"VERIFIED","main_description":"Fictional","source_line_indices":[2]}):
                actual=hybrid.expand_parameter_sources(f"{number}#{bit} {symbol}",results,self.folder)
                self.assertEqual(actual[0]["rrf_score"],0.03)
                self.assertEqual(actual[0]["parameter_source"]["pages"],[10])

if __name__ == "__main__":
    unittest.main()
