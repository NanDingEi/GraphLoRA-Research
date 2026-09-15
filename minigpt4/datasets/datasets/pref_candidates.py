"""Candidate expansion for preference-level distillation (GNN -> LLM).

Wraps a base OOD dataset so that each original sample (user, history, target)
becomes K rows: the real target (label=1) plus K-1 uniformly sampled negative
items (label=0). All rows of a group share the same user/history prompt and
carry the same `cand_group` id, so the model can form the group-wise
preference distributions

    P_L = softmax(yes_logit / tau_L),   P_G = softmax(kappa*cos(z_u,z_c)/tau_G)

and distill the collaborative preference ordering into the LLM's Yes
probability (the collaborative space "raises the generation probability" of
items it considers better matches).
"""
import random

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import default_collate


class PrefCandidateDataset(TorchDataset):
    def __init__(self, base_dataset, num_candidates=2, seed=42):
        assert num_candidates >= 2, "need at least target + 1 negative"
        self.base = base_dataset
        self.num_candidates = int(num_candidates)
        self.rng = random.Random(seed)
        # global item_id -> title map (negatives need titles for the prompt)
        self._title_map = {}
        ann = getattr(base_dataset, "annotation", None)
        if ann is not None:
            for iid, title in zip(ann["TargetItemID"], ann["TargetItemTitle"]):
                self._title_map.setdefault(int(iid), str(title))
        self._num_items = int(ann["TargetItemID"].max()) + 1 if ann is not None else 0
        self._group_counter = 0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        base = dict(self.base[index])
        hist = set(int(x) for x in np.asarray(base["InteractedItemIDs_pad"]).tolist() if int(x) != 0)
        target = int(base["TargetItemID"])
        hist.add(target)
        rows = [base]
        base["cand_group"] = self._group_counter
        seen = set(hist)
        for _ in range(self.num_candidates - 1):
            neg = self.rng.randrange(1, max(self._num_items, 2))
            tries = 0
            while neg in seen and tries < 20:
                neg = self.rng.randrange(1, max(self._num_items, 2))
                tries += 1
            seen.add(neg)
            row = dict(base)
            row["TargetItemID"] = neg
            row["TargetItemTitle"] = self._title_map.get(neg, "unknown movie")
            row["label"] = 0
            row["cand_group"] = self._group_counter
            rows.append(row)
        self._group_counter += 1
        return rows

    def collater(self, samples):
        # samples: list (batch) of lists (K rows) -> flatten, then default_collate
        flat = [row for group in samples for row in group]
        batch = default_collate(flat)
        batch["cand_group"] = torch.tensor([r["cand_group"] for group in samples
                                            for r in group], dtype=torch.long)
        batch["num_groups"] = len(samples)
        return batch

    # delegate attributes the runner/model may touch (collater handled above)
    def __getattr__(self, name):
        return getattr(self.base, name)
