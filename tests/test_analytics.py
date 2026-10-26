import unittest


from backend.analytics import prepare_analytics_dataframe


class TestAnalytics(unittest.TestCase):
    def test_prepare_dataframe_handles_empty_records(self):
        dataframe = prepare_analytics_dataframe([])
        self.assertTrue(dataframe.empty)
        self.assertIn("severity", dataframe.columns)


if __name__ == "__main__":
    unittest.main()
