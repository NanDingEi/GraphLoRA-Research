# GraphLoRA projector regularization — compatibility hotfix v2

This release fixes `forward_v2 must have exactly one return dict ...` when
`forward_v2` has more than one return branch (as in custom N1/RLM versions).

**Only the auto-patcher has changed how it injects the loss:** instead of
editing a specific `return`, it inserts `@projector_regularized_forward` directly
above the `forward_v2` method. The decorator calls the original function and
adds `proj_reg_weight * cosine_reg_loss` to the returned dictionary's `loss`
**only in training mode with gradients enabled**. It keeps other keys and
auxiliary loss terms unchanged. A missing `loss` entry in training raises a
clear error. It does not change the GNN/LoRA injection or create a projector
pretraining stage.

## Installation

At the root of your GraphLoRA repo:

```bash
# v1 failed at --check: --check never modified the model/YAML
unzip -o graphlora_projector_reg_v2.zip
python -m unittest discover -s projector_reg_tests -v
python apply_projector_reg.py \
  --config /home/bbx/user_lx/GraphLoRA/tran_configs/collm_pretrain_mf_ood_scratch_rlm.yaml \
  --weight 0.001 --items 64 --check

# Only after --check succeeds:
python apply_projector_reg.py \
  --config /home/bbx/user_lx/GraphLoRA/tran_configs/collm_pretrain_mf_ood_scratch_rlm.yaml \
  --weight 0.001 --items 64 --apply
python -m py_compile minigpt4/models/minigpt4rec_v2.py
```

The patcher creates timestamped backups before `--apply`. If the current model
has other incompatible changes in its `__init__` or `from_config`, it will
continue to stop rather than guessing how to change them.

## Scientific control

For a single-variable GraphLoRA+projector-reg experiment, disable N1/RLM,
semantic alignment, any dynamic gate, and multilayer injection in the chosen
YAML. Leave `gcn_layer=31`, `gnn_scale=0.1`, legacy injection, and all other
GraphLoRA settings unchanged. Point `--config` to that *clean* experimental
YAML if possible; this hotfix is also compatible with custom-return code but
does not automatically disable custom losses.

`proj_reg_weight: 0` turns the auxiliary loss off at runtime.

## Verification

After applying, check the model diff manually:

```bash
grep -n 'projector_regularized_forward\|proj_reg_weight\|proj_reg_items' \
  minigpt4/models/minigpt4rec_v2.py
```

Expected structure:

```python
@projector_regularized_forward
def forward_v2(self, samples):
    # ALL existing return branches remain untouched
    ...
```

Testing: synthetic CPU unit tests and Python compile, no 7B end-to-end run.
