import unittest

import pandas as pd

from backend.analytics import REPORTLAB_AVAILABLE, generate_pdf_report, prepare_analytics_dataframe


class TestAnalytics(unittest.TestCase):
    def test_prepare_dataframe_handles_empty_records(self):
        dataframe = prepare_analytics_dataframe([])
        self.assertTrue(dataframe.empty)
        self.assertIn("severity", dataframe.columns)

    @unittest.skipUnless(REPORTLAB_AVAILABLE, "reportlab is not installed")
    def test_pdf_report_is_generated(self):
        dataframe = pd.DataFrame([
            {
                "id": 1,
                "date": "2026-07-31",
                "severity": "High",
                "category": "Crash",
                "module": "Image Upload Service",
                "team": "Backend Services",
            }
        ])
        pdf = generate_pdf_report(dataframe)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 500)


if __name__ == "__main__":
    unittest.main()
