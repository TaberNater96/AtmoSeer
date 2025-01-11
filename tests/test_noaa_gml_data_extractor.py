import unittest
from ..etl.\
            extract.\
                noaa_gml_data_extractor import GMLDataExtractor

class TestCO2DataExtractor(unittest.TestCase):

    def test_find_header_end(self):
        sample_data = """
# Example Header
# ------------------------------------------------------------
# DATA
#
# Data starts here
site_code year
MKO 2022
"""
        extractor = GMLDataExtractor(urls=[])
        header_end_idx = extractor.find_header_end(sample_data)
        self.assertEqual(header_end_idx, 6, "Header end index should be 6")

    def test_text_to_dataframe(self):
        sample_data = """
site_code year
MKO 2022
"""
        extractor = GMLDataExtractor(urls=[])
        df = extractor.text_to_dataframe(sample_data, 0)
        self.assertEqual(df.iloc[0]['site_code'], 'MKO')
        self.assertEqual(df.iloc[0]['year'], 2022)

if __name__ == '__main__':
    unittest.main()
