"""G1 CPU tests: python -m unittest tests/test_G1_cf_semantic_graph.py -v."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

from build_G1_cf_semantic_graph import (
    build_graph, graph_normalize, select_semantic_edges, supported_pairs,
)


class G1GraphTests(unittest.TestCase):
    def _inputs(self, home):
        home = Path(home)
        # Three users, six item IDs. Only (item1,item2) has positive ratings
        # from two independent TRAIN users, and matching "Sci-Fi" genres.
        rows = [
            (0, 1, "Space Alpha (2000)", 1),
            (0, 2, "Space Beta (2001)", 1),
            (0, 3, "Love Story (2002)", 0),
            (1, 1, "Space Alpha (2000)", 1),
            (1, 2, "Space Beta (2001)", 1),
            (1, 3, "Love Story (2002)", 0),
            (2, 3, "Love Story (2002)", 1),
            (2, 4, "Dance Time (2003)", 0),
        ]
        df = pd.DataFrame(rows, columns=["uid", "iid", "title", "label"])
        df.to_pickle(home / "train_ood2.pkl")
        rr, cc = [], []
        for u, i, _, y in rows:
            if y:
                rr.extend((u, 3 + i))
                cc.extend((3 + i, u))
        raw = sp.csr_matrix(
            (np.ones(len(rr), dtype=np.float64), (rr, cc)), shape=(9, 9)
        )
        raw.data[:] = 1
        sp.save_npz(home / "movie_graph_train.npz", graph_normalize(raw))
        (home / "movies.dat").write_text(
            "999::Space Alpha (2000)::Sci-Fi|Action\n"
            "998::Space Beta (2001)::Sci-Fi|Action\n"
            "997::Love Story (2002)::Romance\n"
            "996::Dance Time (2003)::Musical\n", encoding="latin-1",
        )
        return raw

    def test_graph_matches_cf_baseline_and_adds_train_supported_item_edges(self):
        with TemporaryDirectory() as d:
            raw = self._inputs(d)
            out = Path(d) / "out" / "G1.npz"
            report = build_graph(d, out, 3, 6)
            self.assertEqual(report["undirected_new_item_edges"], 1)
            self.assertEqual(report["genres_mapped"], 4)
            self.assertEqual(report["train_rows"], 8)
            self.assertLess(report["baseline_max_reconstruction_err"], 1e-5)
            graph = sp.load_npz(out).tocsr()
            self.assertEqual(graph.shape, (9, 9))
            self.assertGreater(float(graph[4, 5]), 0.0)
            self.assertAlmostEqual(float(graph[4, 5]), float(graph[5, 4]), 7)
            self.assertEqual(float(graph[4, 6]), 0.0)
            self.assertEqual(float(graph[5, 6]), 0.0)
            self.assertTrue(np.isfinite(graph.data).all())
            report2 = json.loads(out.with_suffix(".json").read_text())
            self.assertEqual(report2["graph_sha256"], report["graph_sha256"])
            self.assertEqual(report["baseline_nnz"], raw.nnz)

    def test_only_distinct_positive_train_users_supply_support(self):
        positives = {0: {1, 2}, 1: {1, 2}, 2: {1, 3}}
        pairs = supported_pairs(positives)
        self.assertEqual(pairs[(1, 2)], 2)
        self.assertEqual(pairs[(1, 3)], 1)
        self.assertNotIn((2, 3), pairs)
        titles = {1: "Space Alpha (2000)", 2: "Space Beta (2001)",
                  3: "Space Gamma (2002)"}
        genres = {v: frozenset(["sci-fi"]) for v in titles.values()}
        add, stats = select_semantic_edges(
            titles, positives, genres, num_users=3, num_items=6,
            min_shared_users=2,
        )
        self.assertEqual(stats["undirected_new_item_edges"], 1)
        self.assertGreater(float(add[4, 5]), 0.0)
        self.assertEqual(float(add[4, 6]), 0.0)

    def test_fails_before_training_for_wrong_original_cf_normalization(self):
        with TemporaryDirectory() as d:
            self._inputs(d)
            path = Path(d) / "movie_graph_train.npz"
            a = sp.load_npz(path).tocsr()
            a.data *= 0.5
            sp.save_npz(path, a)
            with self.assertRaisesRegex(ValueError, "NOT D"):
                build_graph(d, Path(d) / "G1.npz", 3, 6)


if __name__ == "__main__":
    unittest.main()
