import datetime as dt
import unittest

import numpy as np

from expdb.config import compute_config_hash
from expdb.db import DatabaseError, Result, _encode, get_conn, temporary_database


class TestConfigHash(unittest.TestCase):
    def test_config_hash_stability(self):
        c1 = {"method": "reldec", "z": 1, "alpha": 0.1, "workers": 40, "max_frames": 100}
        c2 = {"method": "reldec", "alpha": 0.1, "z": 1, "workers": 1, "max_frames": 10000}

        # Hash should ignore workers and max_frames, and order shouldn't matter
        self.assertEqual(compute_config_hash(c1), compute_config_hash(c2))

        # Changing a core param should change the hash
        c3 = {"method": "reldec", "z": 1, "alpha": 0.2, "workers": 40, "max_frames": 100}
        self.assertNotEqual(compute_config_hash(c1), compute_config_hash(c3))


class TestWireFormat(unittest.TestCase):
    """Encoding/decoding of the Neon HTTPS client, no network needed."""

    def test_encode(self):
        self.assertEqual(_encode(np.int64(7)), "7")
        self.assertEqual(_encode(np.float64(0.1)), "0.1")
        self.assertEqual(_encode(True), "true")
        self.assertIsNone(_encode(None))
        with self.assertRaises(ValueError):
            _encode(dt.datetime(2026, 1, 1))  # naive datetimes are refused

    def test_unknown_type_raises(self):
        with self.assertRaises(TypeError):
            Result({"fields": [{"name": "x", "dataTypeID": 114}], "rows": [["{}"]]})  # json

    def test_timestamp_parsing(self):
        r = Result({"fields": [{"name": "t", "dataTypeID": 1184}],
                    "rows": [["2026-09-15 18:27:48.24+00"]]})
        self.assertEqual(r.fetchone()[0],
                         dt.datetime(2026, 9, 15, 18, 27, 48, 240000, tzinfo=dt.timezone.utc))


class TestExpDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._db = temporary_database()
        cls._db.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._db.__exit__(None, None, None)

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

    def test_commit_without_row_raises(self):
        from expdb import get_or_create_config, commit_chunk
        config_id = get_or_create_config({"method": "no_row"})
        with self.assertRaises(RuntimeError):
            commit_chunk(config_id, 1.0, 100, 1000, {"frames": 1})

    def test_float_snr_keys_round_trip(self):
        # snr_db is part of the primary key and the coverage dict key, so it
        # must come back bit-identical.
        from expdb import get_or_create_config, ensure_eval_row, get_coverage
        config_id = get_or_create_config({"method": "floats"})
        snrs = [0.1, 1.5, 2.675, 1 / 3, np.float64(0.30000000000000004)]
        for s in snrs:
            ensure_eval_row(config_id, s, 10, 10)
        self.assertEqual(set(get_coverage(config_id, 10, 10)), {float(s) for s in snrs})

    def test_bigint_counters(self):
        from expdb import get_or_create_config, ensure_eval_row, commit_chunk, get_eval_row
        config_id = get_or_create_config({"method": "big"})
        ensure_eval_row(config_id, 0.0, 10**9, 10**9)
        big = 3 * 2**31
        commit_chunk(config_id, 0.0, 10**9, 10**9,
                     {"frames": np.int64(5), "bit_errors": 1, "total_bits": big, "frame_errors": 0, "messages": big})
        row = get_eval_row(config_id, 0.0, 10**9, 10**9)
        self.assertEqual((row["total_bits"], row["messages"], row["frames_done"]), (big, big, 5))

    def test_query_ber(self):
        from expdb import get_or_create_config, ensure_eval_row, commit_chunk, query_ber
        config_id = get_or_create_config({"method": "ber"})
        ensure_eval_row(config_id, 1.0, 5, 100)
        commit_chunk(config_id, 1.0, 5, 100,
                     {"frames": 40, "bit_errors": 3, "total_bits": 4000, "frame_errors": 2, "messages": 800})
        (r,) = query_ber(config_id, 5, 100)
        self.assertEqual(r["ber"], 3 / 4000)
        self.assertEqual(r["fer"], 2 / 40)
        self.assertEqual(r["avg_messages"], 800 / 40)

    def test_run_lifecycle(self):
        from expdb import (get_or_create_config, create_run, update_run_status, set_checkpoint,
                           add_intermediate_checkpoint, get_latest_checkpoint)
        config_id = get_or_create_config({"method": "runs"})
        self.assertIsNone(get_latest_checkpoint(config_id))
        run_id = create_run(config_id, "train", {"method": "runs", "workers": 4})
        add_intermediate_checkpoint(run_id, "a.json")
        add_intermediate_checkpoint(run_id, "b.json")
        set_checkpoint(run_id, "final.json", 100)
        update_run_status(run_id, "completed")
        self.assertEqual(get_latest_checkpoint(config_id), "final.json")

        row = get_conn().execute(
            "SELECT intermediate_checkpoints, completed_at, status FROM runs WHERE run_id = $1",
            (run_id,)).fetchone()
        self.assertEqual(row[0], '["a.json", "b.json"]')
        self.assertIsNotNone(row[1].tzinfo)
        self.assertEqual(row[2], "completed")

    def test_extend_eval(self):
        from expdb import get_or_create_config, ensure_eval_row, commit_chunk, extend_eval, get_eval_row
        config_id = get_or_create_config({"method": "extend"})
        ensure_eval_row(config_id, 2.0, 10**6, 50)
        commit_chunk(config_id, 2.0, 10**6, 50,
                     {"frames": 50, "bit_errors": 7, "total_bits": 500, "frame_errors": 3, "messages": 9})
        extend_eval(config_id, 2.0, 10**6, 50, 200)
        row = get_eval_row(config_id, 2.0, 10**6, 200)
        self.assertEqual((row["frames_done"], row["bit_errors"], row["completed"]), (50, 7, False))

    def test_sql_errors_raise(self):
        with self.assertRaises(DatabaseError) as cm:
            get_conn().execute("SELECT * FROM no_such_table")
        self.assertEqual(cm.exception.code, "42P01")


if __name__ == '__main__':
    unittest.main()
