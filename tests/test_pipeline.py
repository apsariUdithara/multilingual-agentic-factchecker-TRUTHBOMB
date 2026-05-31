import unittest
from src.tools.traffic_db_tool import verify_traffic_law_claim

class TestPipeline(unittest.TestCase):
    def test_traffic_tool_exists(self):
        self.assertIsNotNone(verify_traffic_law_claim)

    def test_traffic_query(self):
        # This test will attempt to load the model and DB.
        # Use a simple query to verify integration.
        claim = "What is the fine for speeding in Sri Lanka?"
        # Note: This might be slow on the first run due to model loading.
        try:
            result = verify_traffic_law_claim.invoke({"claim": claim})
            print(f"Traffic Tool Result: {result}")
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)
        except Exception as e:
            self.fail(f"Traffic tool failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
