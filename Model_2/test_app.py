import unittest
from fastapi.testclient import TestClient
import app_copy  # Your app file with FastAPI app
import re
import numpy as np

def convert_np_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_np_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_floats(v) for v in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    return obj

class TestSearchEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app_copy.app)
        if app_copy.dataset is None:
            app_copy.load_and_index_dataset()

    def test_search_endpoint(self):
        response = self.client.get("/search", params={"q": "shirt repair"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertLessEqual(len(data), 10)
        if data:
            first = data[0]
            self.assertIn("Service", first)
            self.assertIn("similarity_score", first)
            self.assertIn("match_type", first)
            self.assertIn("match_detail", first)
            self.assertIn("search_terms", first)

    def test_debug_search_endpoint(self):
        response = self.client.get("/debug_search", params={"q": "shirt repair"})
        self.assertEqual(response.status_code, 200)
        raw_data = response.json()
        data = convert_np_floats(raw_data)
        self.assertIn("query_processing", data)
        self.assertIn("sample_entries", data)

    def test_search_strategy_endpoint(self):
        response = self.client.get("/search_strategy")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("search_strategy", data)
        self.assertIn("stages", data)
        self.assertIsInstance(data["stages"], list)

    def test_correct_query_spell_correction(self):
        misspelled = "shrit repar"
        corrected = app_copy.correct_query(misspelled)
        self.assertNotEqual(misspelled, corrected)
        self.assertTrue("shirt" in corrected.lower() or "repair" in corrected.lower())

    def test_lemmatize_and_lower(self):
        text = "Shirts are repaired"
        normalized = app_copy.lemmatize_and_lower(text)
        self.assertIsInstance(normalized, str)
        self.assertIn("shirt", normalized)
        self.assertIn("repair", normalized)

    def test_extract_price(self):
        self.assertEqual(app_copy.extract_price("The price is 150 dollars"), 150)
        self.assertEqual(app_copy.extract_price("Cost: 99"), 99)
        self.assertIsNone(app_copy.extract_price("No price here"))

    def test_search_price_filter(self):
        query = "shirt repair 150"
        response = self.client.get("/search", params={"q": query})
        self.assertEqual(response.status_code, 200)
        results = response.json()
        for r in results:
            try:
                price = float(r.get("Price", 0))
                self.assertTrue(abs(price - 150) <= 50)
            except (ValueError, TypeError):
                self.fail(f"Price missing or invalid in result: {r.get('Price')}")

    def test_search_reflects_corrected_query(self):
        misspelled = "shrit repar"
        response = self.client.get("/search", params={"q": misspelled})
        self.assertEqual(response.status_code, 200)
        results = response.json()
        if results:
            for r in results:
                terms = r.get("search_terms", [])
                self.assertTrue(any("shirt" in term or "repair" in term for term in terms))

    def test_search_keyword_vs_semantic_prioritization(self):
        query = "leather jacket repair"
        response = self.client.get("/search", params={"q": query})
        self.assertEqual(response.status_code, 200)
        results = response.json()
        self.assertTrue(len(results) > 0)
        match_types = [r.get("match_type") for r in results]
        if "exact_service" in match_types:
            first_exact_index = match_types.index("exact_service")
            self.assertLess(first_exact_index, len(match_types))
        semantic_indices = [i for i, m in enumerate(match_types) if m == "semantic"]
        if semantic_indices:
            max_non_semantic = max([i for i, m in enumerate(match_types) if m != "semantic"], default=-1)
            self.assertTrue(all(i > max_non_semantic for i in semantic_indices))

if __name__ == "__main__":
    unittest.main()
