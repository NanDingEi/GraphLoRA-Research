"""CPU-only tests for G3 popularity weighting."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
import scipy.sparse as sp
import numpy as np

from build_G3_popularity_graph import build_graph
from build_G1_cf_semantic_graph import graph_normalize


class G3Tests(unittest.TestCase):
    def setup_graph(self, d):
        d = Path(d)
        # 3 users and 4 items. item1 popular (3 users), item2 rare (1).
        edges = [(0,1),(1,1),(2,1),(0,2),(2,3)]
        df = pd.DataFrame(
            [(u,i,"Movie {}".format(i),1) for u,i in edges] +
            [(1,2,"Movie 2",0)],
            columns=["uid","iid","title","label"]
        )
        df.to_pickle(d / "train_ood2.pkl")
        rows = [u for u,i in edges] + [3+i for u,i in edges]
        cols = [3+i for u,i in edges] + [u for u,i in edges]
        mat = sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(7,7))
        sp.save_npz(d / "movie_graph_train.npz", graph_normalize(mat))
        return edges

    def test_only_reweight_original_train_edges_and_normalize(self):
        with TemporaryDirectory() as d:
            edges = self.setup_graph(d)
            out=Path(d)/"g3.npz"
            report=build_graph(d,out,3,4)
            graph=sp.load_npz(out).tocsr()
            base=sp.load_npz(Path(d)/"movie_graph_train.npz").tocsr()
            self.assertEqual(report["train_positive_edges"], len(edges))
            self.assertEqual(graph.nnz,base.nnz)
            self.assertEqual(set(zip(*graph.nonzero())),set(zip(*base.nonzero())))
            self.assertAlmostEqual(float(graph[3+1,0]),float(graph[0,3+1]),places=6)
            self.assertTrue(np.isfinite(graph.data).all())
            self.assertTrue(out.with_suffix(".json").is_file())
            self.assertGreater(float(report["max_actual_weight"]),
                               float(report["min_actual_weight"]))

    def test_reject_leaked_graph_edge(self):
        with TemporaryDirectory() as d:
            self.setup_graph(d)
            g=sp.load_npz(Path(d)/"movie_graph_train.npz").tolil()
            g[1,5]=1
            g[5,1]=1
            sp.save_npz(Path(d)/"movie_graph_train.npz",graph_normalize(g.tocsr()))
            with self.assertRaisesRegex(ValueError,"not matching"):
                build_graph(d,Path(d)/"g3.npz",3,4)


if __name__ == "__main__":
    unittest.main()
