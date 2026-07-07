import os
import unittest
import json
import duckdb
from expdb.config import compute_config_hash, normalize_config, HASH_EXCLUSIONS
from expdb.db import DB_PATH, get_conn, _local

class TestExpDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use an in-memory DB for tests
        if hasattr(_local, "conn"):
            _local.conn.close()
            del _local.conn
        _local.conn = duckdb.connect(':memory:')
        from expdb.db import _init_schema
        _init_schema(_local.conn)
        
    def test_config_hash_stability(self):
        c1 = {"method": "reldec", "z": 1, "alpha": 0.1, "workers": 40, "max_frames": 100}
        c2 = {"method": "reldec", "alpha": 0.1, "z": 1, "workers": 1, "max_frames": 10000}
        
        # Hash should ignore workers and max_frames, and order shouldn't matter
        self.assertEqual(compute_config_hash(c1), compute_config_hash(c2))
        
        # Changing a core param should change the hash
        c3 = {"method": "reldec", "z": 1, "alpha": 0.2, "workers": 40, "max_frames": 100}
        self.assertNotEqual(compute_config_hash(c1), compute_config_hash(c3))

    def test_db_eval_flow(self):
        from expdb import get_or_create_config, ensure_eval_row, commit_chunk, get_eval_row, get_coverage
        
        config = {"method": "reldec", "z": 1}
        config_id = get_or_create_config(config)
        
        snr = 2.0
        ensure_eval_row(config_id, snr, 100, 10000)
        
        # Commit a chunk
        commit_chunk(config_id, snr, 100, 10000, {"frames": 50, "bit_errors": 5, "total_bits": 500, "frame_errors": 1})
        
        row = get_eval_row(config_id, snr, 100, 10000)
        self.assertEqual(row["frames_done"], 50)
        self.assertEqual(row["completed"], False)
        
        # Commit another chunk that hits target_frame_errors
        commit_chunk(config_id, snr, 100, 10000, {"frames": 50, "bit_errors": 5, "total_bits": 500, "frame_errors": 99})
        
        row = get_eval_row(config_id, snr, 100, 10000)
        self.assertEqual(row["frames_done"], 100)
        self.assertEqual(row["completed"], True)
        
        cov = get_coverage(config_id, 100, 10000)
        self.assertEqual(cov[snr]["frames_done"], 100)

if __name__ == '__main__':
    unittest.main()
