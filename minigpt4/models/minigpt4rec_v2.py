import logging
import random
from minigpt4.models.projector_structure_reg import projector_regularized_forward
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.sparse as sp
from minigpt4.common.registry import registry
from peft.tuners.lora import Linear as OriginalLinear
from minigpt4.models.rec_model import Rec2Base, disabled_train
from peft.utils import transpose
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, GenerationConfig
import re
import numpy as np
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, \
    set_peft_model_state_dict


def get_ids_order(prompt):
    id_flags = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]
    id_order_ = []
    for flag_ in id_flags:
        pos_ = prompt.find(flag_)
        if pos_ >= 0:
            id_order_.append(pos_)
    id_order_ = np.argsort(np.array(id_order_))
    return id_order_


# GCN Layer Definition
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(GCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, adj_matrix, node_features, node_ids=None):
        adj_matrix = adj_matrix.to(dtype=torch.float32, device=self.device)
        node_features = node_features.to(dtype=torch.float32, device=self.device)
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense().to(dtype=torch.float32)
        row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum, dtype=torch.float32))
        d_mat_inv_sqrt = torch.diag(torch.pow(row_sum, -0.5)).to(dtype=torch.float32, device=self.device)
        norm_adj = torch.sparse.mm(d_mat_inv_sqrt, torch.sparse.mm(adj_matrix, d_mat_inv_sqrt)).to(dtype=torch.float32,
                                                                                                   device=self.device)
        support = self.linear(node_features)
        support = self.dropout(support)
        output = torch.sparse.mm(norm_adj, support)
        return output


class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(TwoLayerGCN, self).__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, dropout)

    def forward(self, adj_matrix, node_features, node_ids=None):
        h = self.gcn1(adj_matrix, node_features, node_ids)
        h = F.relu(h)
        h = self.gcn2(adj_matrix, h, node_ids)
        return h


class LightGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=3, dropout=0.1):
        super(LightGCN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 仅使用一层线性变换将初始嵌入映射到输出维度
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, adj_matrix, node_features, node_ids=None, return_penult=False):
        adj_matrix = adj_matrix.to(dtype=torch.float32, device=self.device)
        node_features = node_features.to(dtype=torch.float32, device=self.device)

        # 计算归一化邻接矩阵的度数
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum, dtype=torch.float32))
        d_mat_inv_sqrt = torch.diag(torch.pow(row_sum, -0.5)).to(dtype=torch.float32, device=self.device)
        norm_adj = torch.sparse.mm(d_mat_inv_sqrt, torch.sparse.mm(adj_matrix, d_mat_inv_sqrt))

        # 多层传播
        embeddings = [node_features]
        h = node_features
        for _ in range(self.num_layers):
            h = torch.sparse.mm(norm_adj, h)
            h = self.dropout(h)
            embeddings.append(h)

        # 平均所有层的嵌入
        final_emb = torch.mean(torch.stack(embeddings, dim=0), dim=0)

        # 映射到目标维度
        output = self.linear(final_emb)
        output = F.normalize(output, dim=-1) * 0.1  # 归一化并缩放
        output = output.to(dtype=torch.float16)
        if return_penult:
            # 256-d layer-mean embedding before the linear projection to out_dim
            penult = F.normalize(final_emb, dim=-1)
            return output, penult
        return output


class NGCFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(NGCFLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        # 两个线性变换：W1 用于自身和邻居，W2 用于交互项
        self.W1 = nn.Linear(in_dim, out_dim)
        self.W2 = nn.Linear(in_dim, out_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn.init.xavier_uniform_(self.W1.weight, gain=nn.init.calculate_gain('leaky_relu', param=0.2))
        nn.init.xavier_uniform_(self.W2.weight, gain=nn.init.calculate_gain('leaky_relu', param=0.2))
        if self.W1.bias is not None:
            nn.init.zeros_(self.W1.bias)
        if self.W2.bias is not None:
            nn.init.zeros_(self.W2.bias)

    def forward(self, adj_matrix, node_features):
        adj_matrix = adj_matrix.to(dtype=torch.float32, device=self.device)
        node_features = node_features.to(dtype=torch.float32, device=self.device)

        # 计算归一化系数
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        col_sum = torch.sparse.sum(adj_matrix, dim=0).to_dense()
        row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum, dtype=torch.float32))
        col_sum = torch.where(col_sum > 0, col_sum, torch.ones_like(col_sum, dtype=torch.float32))
        d_mat_inv_sqrt_row = torch.diag(torch.pow(row_sum, -0.5))
        d_mat_inv_sqrt_col = torch.diag(torch.pow(col_sum, -0.5))
        norm_adj = torch.sparse.mm(d_mat_inv_sqrt_row, torch.sparse.mm(adj_matrix, d_mat_inv_sqrt_col))

        # 自身变换
        h_self = self.W1(node_features)

        # 邻居聚合
        h_neigh = torch.sparse.mm(norm_adj, node_features)
        h_neigh = self.W1(h_neigh)

        # 交互项（逐元素乘法）
        h_inter = torch.sparse.mm(norm_adj, node_features * node_features)
        h_inter = self.W2(h_inter)

        # 合并并激活
        output = F.leaky_relu(h_self + h_neigh + h_inter, negative_slope=0.2)
        output = self.dropout(output)
        return output


class NGCF(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.1):
        super(NGCF, self).__init__()
        self.layers = nn.ModuleList([NGCFLayer(in_dim if i == 0 else hidden_dim,
                                               hidden_dim if i < num_layers - 1 else out_dim,
                                               dropout) for i in range(num_layers)])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, adj_matrix, node_features, node_ids=None, return_penult=False,
                pre_last=None):
        # pre_last: optional callable applied to the raw penultimate hidden
        # state right before the last (8-d) layer. Used by the Anchor-
        # Preserving Semantic Refinement adapter: h* = h + alpha*Delta(h).
        # Returns (output, penult, penult_star) when return_penult.
        h = node_features
        penult = None
        penult_star = None
        for i, layer in enumerate(self.layers):
            if return_penult and i == len(self.layers) - 1:
                # 256-d embedding feeding the last (8-d) layer, before the
                # final normalize*0.1 / fp16 cast used for LoRA injection
                penult_uncorr = F.normalize(h, dim=-1)
                if pre_last is not None:
                    h, penult_star = pre_last(h, penult_uncorr)
                else:
                    penult_star = None
                penult = penult_uncorr
            h = layer(adj_matrix, h)
        output = F.normalize(h, dim=-1) * 0.1
        output = output.to(dtype=torch.float16)
        if return_penult:
            return output, penult, penult_star
        return output


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, sample_size=25):
        super(GraphSAGE, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.sample_size = sample_size
        self.dropout = nn.Dropout(dropout)

        self.linear_self1 = nn.Linear(in_dim, hidden_dim)
        self.linear_neigh1 = nn.Linear(in_dim, hidden_dim)
        self.linear_self2 = nn.Linear(hidden_dim, out_dim)
        self.linear_neigh2 = nn.Linear(hidden_dim, out_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn.init.xavier_uniform_(self.linear_self1.weight)
        nn.init.xavier_uniform_(self.linear_neigh1.weight)
        nn.init.xavier_uniform_(self.linear_self2.weight)
        nn.init.xavier_uniform_(self.linear_neigh2.weight)

    def forward(self, adj_matrix, node_features, node_ids=None):
        adj_matrix = adj_matrix.to(dtype=torch.float32, device=self.device)
        node_features = node_features.to(dtype=torch.float32, device=self.device)

        # 第一层：聚合邻居特征
        if self.training:
            sampled_indices = []
            for i in range(adj_matrix.shape[0]):
                neighbors = torch.nonzero(adj_matrix[i], as_tuple=False).squeeze(-1)
                if len(neighbors) > self.sample_size:
                    neighbors = neighbors[torch.randperm(len(neighbors))[:self.sample_size]]
                sampled_indices.append(neighbors)
        else:
            sampled_indices = [torch.nonzero(adj_matrix[i], as_tuple=False).squeeze(-1) for i in
                               range(adj_matrix.shape[0])]

        h_neigh = torch.zeros_like(node_features)
        for i in range(node_features.shape[0]):
            if len(sampled_indices[i]) > 0:
                h_neigh[i] = node_features[sampled_indices[i]].mean(dim=0)

        h_self = self.linear_self1(node_features)
        h_neigh = self.linear_neigh1(h_neigh)
        h = F.relu(h_self + h_neigh)
        h = self.dropout(h)

        # 第二层变换
        h_neigh = torch.zeros_like(h)
        for i in range(h.shape[0]):
            if len(sampled_indices[i]) > 0:
                h_neigh[i] = h[sampled_indices[i]].mean(dim=0)
        h_self = self.linear_self2(h)
        h_neigh = self.linear_neigh2(h_neigh)
        output = h_self + h_neigh
        output = F.normalize(output, dim=-1) * 0.1
        output = output.to(dtype=torch.float16)
        return output


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

        self.linear = nn.Linear(in_dim, out_dim)
        self.attention = nn.Linear(2 * out_dim, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, adj_matrix, node_features):
        node_features = node_features.to(dtype=torch.float32, device=self.device)
        h = self.linear(node_features)
        h = self.dropout(h)

        N = h.shape[0]
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_dim)
        e = F.leaky_relu(self.attention(a_input), negative_slope=self.alpha)
        e = e.squeeze(-1)

        adj_matrix = adj_matrix.to(dtype=torch.float32, device=self.device)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, h)
        return h_prime


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, alpha=0.2):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, dropout, alpha)
        self.gat2 = GATLayer(hidden_dim, out_dim, dropout, alpha)

    def forward(self, adj_matrix, node_features, node_ids=None):
        h = self.gat1(adj_matrix, node_features)
        h = F.elu(h)
        h = self.gat2(adj_matrix, h)
        output = F.normalize(h, dim=-1) * 0.1
        output = output.to(dtype=torch.float16)
        return output



class MyLinear(OriginalLinear):
    def __init__(self, original_linear, layer_idx=None, lora_scale=1.0, gnn_scale=0, gcn_layer=0, collab_mode="legacy", module_tag="q"):
        super().__init__(
            adapter_name=original_linear.active_adapter,
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            fan_in_fan_out=original_linear.fan_in_fan_out,
            is_target_conv_1d_layer=original_linear.is_target_conv_1d_layer
        )
        self.weight = original_linear.weight
        self.lora_A = original_linear.lora_A
        self.lora_B = original_linear.lora_B
        self.lora_dropout = original_linear.lora_dropout
        self.scaling = original_linear.scaling
        self.merged = original_linear.merged
        self.lora_scale = lora_scale
        self.gnn_scale = gnn_scale
        self.layer_idx = layer_idx
        self.gcn_layer = gcn_layer
        # COLLAB_INJECTION_V1
        self.collab_mode = collab_mode
        self.module_tag = module_tag
        if collab_mode not in ("legacy", "v_only", "gated_l31", "conditional_lora"):
            raise ValueError(f"Unknown collab_mode: {collab_mode}")
        rank = self.lora_A[self.active_adapter].out_features
        # The collaborative signal is injected at historical item-token positions.
        # Under causal attention, Q at those prefix positions does not directly
        # affect later target-token predictions; K/V do. Create trainable
        # collaborative modules only for V, avoiding permanently dead Q gates.
        if collab_mode == "gated_l31" and self.module_tag == "v":
            self.collab_gate = nn.Linear(2 * rank, 1)
            nn.init.zeros_(self.collab_gate.weight)
            nn.init.zeros_(self.collab_gate.bias)
        elif collab_mode == "conditional_lora" and self.module_tag == "v":
            self.collab_conditioner = nn.Linear(rank, rank)
            nn.init.zeros_(self.collab_conditioner.weight)
            nn.init.zeros_(self.collab_conditioner.bias)

        self.gcn_output = None
        self.batch_indices = None
        self.token_positions = None
        self.local_node_ids = None
        # rank-wise pair modulation (optional): per-batch-row gate in R^r,
        # zero-init MLP => sigmoid(0)*2 = 1 => identity at start
        self.rel_pair_gate = None
        # A0 injection diagnostics (config-gated, training only): per-step
        # stats of the LoRA-A component vs the injected GNN component in the
        # bottleneck, BEFORE lora_B. Pure no-grad bookkeeping — consumes no
        # RNG and does not touch any tensor in the training graph.
        self.inj_diag = False
        # Preserve module_tag passed by the constructor ("q" or "v").
        # Overwriting it here makes the V-only collaborative branch unreachable.
        self._diag_buf = []
        self._diag_step = 0
        # A1 injection warm-up factor; stays 1.0 unless forward_v2 overrides
        # it (training only, config-gated). Eval always uses 1.0.
        self.warmup_factor = 1.0

    def _log_inj_diag(self, a, g):
        """a = lora_scale * LoRA-A output, g = gnn_scale * z_G, both at the
        injection positions. Records ||a||, ||g||, R_inj, C_inj and prints a
        windowed summary every 100 training steps."""
        with torch.no_grad():
            af = a.float()
            gf = g.float()
            na = af.norm(dim=-1)
            ng = gf.norm(dim=-1)
            r = ng / na.clamp_min(1e-8)
            c = (af * gf).sum(-1) / (na * ng).clamp_min(1e-8)
        self._diag_buf.append((r, c, na.mean(), ng.mean()))
        if len(self._diag_buf) >= 100:
            r_all = torch.cat([x[0] for x in self._diag_buf])
            c_all = torch.cat([x[1] for x in self._diag_buf])
            na_m = torch.stack([x[2] for x in self._diag_buf]).mean().item()
            ng_m = torch.stack([x[3] for x in self._diag_buf]).mean().item()
            print(
                f"[inj-diag][{self.module_tag}] steps={self._diag_step - len(self._diag_buf) + 1}-{self._diag_step} "
                f"R_med={r_all.median().item():.4f} R_p90={r_all.quantile(0.9).item():.4f} "
                f"C_med={c_all.median().item():.4f} C_p10={c_all.quantile(0.1).item():.4f} "
                f"|a|={na_m:.4f} |g|={ng_m:.4f}",
                flush=True
            )
            self._diag_buf = []

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        weight = transpose(self.weight, self.fan_in_fan_out)
        result = F.linear(x, weight, bias=self.bias)

        if self.active_adapter not in self.lora_A.keys():
            return result.to(previous_dtype)

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            return result.to(previous_dtype)

        if self.r[self.active_adapter] > 0 and not self.merged:
            x_lora = x.to(self.lora_A[self.active_adapter].weight.dtype)
            x_lora = self.lora_dropout[self.active_adapter](x_lora)
            lora_a_output = self.lora_A[self.active_adapter](x_lora)

            if self.layer_idx == self.gcn_layer and self.gcn_output is not None:
                valid_mask = self.local_node_ids != -1
                if valid_mask.any():
                    batch_indices = self.batch_indices[valid_mask]
                    token_positions = self.token_positions[valid_mask]
                    local_node_ids = self.local_node_ids[valid_mask]

                    if self.training and self.inj_diag:
                        self._diag_step += 1
                        # g must include the effective injection scale
                        # (gnn_scale * warmup_factor) so R_inj matches
                        # ||beta(t) z_G|| / ||a|| under warm-up too
                        self._log_inj_diag(
                            lora_a_output[batch_indices, token_positions],
                            self.gcn_output[local_node_ids]
                            * (self.gnn_scale * self.warmup_factor),
                        )

                    if self.collab_mode == "legacy" or (
                        self.collab_mode == "v_only" and self.module_tag == "v"
                    ):
                        # V-only ablation: execute exactly the legacy injection
                        # expression on V, while Q remains ordinary LoRA.
                        # No gate parameters or extra scaling are introduced.
                        # Original GraphLoRA injection:
                        # r' = lora_scale * r + gnn_scale * z_GNN
                        # A1 warm-up: effective gnn_scale = gnn_scale * warmup_factor.
                        # The factor==1.0 branch keeps the ORIGINAL expression so the
                        # default path stays bitwise identical to I3.
                        if self.warmup_factor == 1.0:
                            lora_a_output[batch_indices, token_positions] = (
                                lora_a_output[batch_indices, token_positions] * self.lora_scale
                                + self.gcn_output[local_node_ids] * self.gnn_scale
                            )
                        else:
                            lora_a_output[batch_indices, token_positions] = (
                                lora_a_output[batch_indices, token_positions] * self.lora_scale
                                + self.gcn_output[local_node_ids]
                                * (self.gnn_scale * self.warmup_factor)
                            )
    
                    elif self.collab_mode in ("gated_l31", "conditional_lora"):
                        # Only V receives the collaborative signal at historical
                        # item-token positions. Q's prefix-position output is
                        # causally disconnected from later target-token losses.
                        if self.module_tag == "v":
                            a = lora_a_output[batch_indices, token_positions]
                            z = self.gcn_output[local_node_ids].to(dtype=a.dtype)
                            if z.shape != a.shape:
                                raise RuntimeError(f"GNN/LoRA rank mismatch: {z.shape} vs {a.shape}")
                            if self.collab_mode == "gated_l31":
                                gate_in = torch.cat((a.float(), z.float()), dim=-1)
                                gate = (2.0 * torch.sigmoid(self.collab_gate(gate_in))).to(dtype=a.dtype)
                                updated = (a * self.lora_scale +
                                           z * (self.gnn_scale * self.warmup_factor) * gate)
                            else:
                                modulation = (2.0 * torch.sigmoid(
                                    self.collab_conditioner(z.float()))).to(dtype=a.dtype)
                                updated = a * self.lora_scale * modulation
                            lora_a_output[batch_indices, token_positions] = updated
            if self.rel_pair_gate is not None:
                # rank-wise pair modulation: g = 2*sigmoid(W c_ui) in (0,2),
                # zero-init W => g == 1 (identity). Modulates the LoRA-A
                # output at the ID-token positions per rank dimension.
                g = (2.0 * torch.sigmoid(
                    self.rel_pair_gate[batch_indices]
                )).to(lora_a_output.dtype)
                lora_a_output[batch_indices, token_positions] = (
                    lora_a_output[batch_indices, token_positions] * g
                )

            lora_output = self.lora_B[self.active_adapter](lora_a_output)
            result += lora_output

        return result.to(previous_dtype)

@registry.register_model("mini_gpt4rec_v2")
class MiniGPT4Rec_v2(Rec2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4rec.yaml",
    }

    def __init__(
            self,
            rec_model="MF",
            rec_config=None,
            pretrained_rec=None,
            freeze_rec=True,
            rec_precision='fp16',
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,
            device_8bit=0,
            proj_token_num=1,
            proj_drop=0,
            lora_config=None,
            proj_mid=5,
            freeze_lora=False,
            freeze_proj=False,
            lora_gnn="lightgcn",
            dataset="book",
            lora_scale=1.0,
            gnn_scale=0,
            gcn_layer=0,
            collab_mode="legacy",
            gnn_semantic_align=False,
            gnn_semantic_weight=0.05,
            train_gnn_with_rec_loss=True,
            pref_distill=False,
            pref_distill_weight=0.5,
            pref_tau_l=1.0,
            pref_tau_g=1.0,
            pref_kappa_init=5.0,
            gnn_conf_loss_weight=0.0,
            rel_inject=False,
            rel_gate=False,
            rel_rank_mod=False,
            sem_structural_skip=False,
            grad_clip=0.0,
            inj_warmup_start=0,
            inj_warmup_len=0,
            sem_adapter=False,
            sem_alpha=0.0,
            sem_rho=0.05,
            sem_corr_weight=0.04,
            sem_adapter_in_path=True,
            sem_trunk_grad=False,
            sem_decay=False,
            sem_decay_steps="11297,22594",
            inj_diag=False,
            gnn_injection_warmup_ratio=0.0,
            gnn_injection_warmup_total_steps=0,
            sem_consensus=False,
            sem_consensus_layers="6,8,10",
            sem_conf_tau=0.001,
            sem_consensus_target="mean",
            lora_rank_groups=None,
            fix_autocast_cache=False,
            proj_reg_weight=0.0,
            proj_reg_items=64,
    ):
        super().__init__()
        self.low_resource = low_resource
        self.proj_token_num = proj_token_num
        # Optional item-item structure regularization on llama_proj only.
        self.proj_reg_weight = float(proj_reg_weight)
        self.proj_reg_items = int(proj_reg_items)
        self._proj_reg_step = 0
        self._last_proj_reg_loss = None
        self.dataset = dataset

        # Only the requested semantic-correction controls.
        self.gnn_semantic_align = bool(gnn_semantic_align)
        self.gnn_semantic_weight = float(gnn_semantic_weight)
        self.train_gnn_with_rec_loss = bool(train_gnn_with_rec_loss)

        # Preference-level G->L distillation (v12 recipe): the GNN's group-wise
        # preference distribution is distilled into the LLM Yes-probability.
        self.pref_distill = bool(pref_distill)
        self.pref_distill_weight = float(pref_distill_weight)
        self.pref_tau_l = float(pref_tau_l)
        self.pref_tau_g = float(pref_tau_g)
        self.gnn_conf_loss_weight = float(gnn_conf_loss_weight)
        if self.pref_distill or self.gnn_conf_loss_weight > 0:
            self.gnn_conf_kappa = nn.Parameter(torch.tensor(float(pref_kappa_init)))
        # Fixed internal teacher depth: early enough to stay before GraphLoRA
        # injection for the standard gcn_layer=31 setup. It is intentionally
        # not exposed as another training hyper-parameter.
        self._gnn_semantic_layer = min(8, max(int(gcn_layer) - 1, 0))
        self._semantic_step = 0
        # Anchor-Preserving Semantic Refinement flags (user design 09-30).
        # sem_adapter_on replaces the old L_sem with the anchored L_corr;
        # the adapter is zero-init so any alpha/rho starts at the baseline.
        self.sem_adapter_on = bool(sem_adapter)
        self.sem_alpha = float(sem_alpha)
        self.sem_rho = float(sem_rho)
        self.sem_corr_weight = float(sem_corr_weight)
        self.sem_adapter_in_path = bool(sem_adapter_in_path)
        self.sem_trunk_grad = bool(sem_trunk_grad)
        # Transient Semantic Correction (user design 09-30): rho_t decays
        # linearly from sem_rho to 0 over [T1, T2] global steps — early
        # semantic correction, late rec-task dominance
        self.sem_decay = bool(sem_decay)
        _ds = str(sem_decay_steps).split(",")
        self.sem_decay_t1 = int(_ds[0])
        self.sem_decay_t2 = int(_ds[1]) if len(_ds) > 1 else int(_ds[0]) * 2
        # A0/A1 injection controls. Both are pure schedules/bookkeeping: no new
        # parameters, no RNG consumption. ratio=0 keeps the training path
        # bitwise identical to the I3 champion.
        if collab_mode not in ("legacy", "v_only", "gated_l31", "conditional_lora"):
            raise ValueError(f"Unknown collab_mode: {collab_mode}")
        self.collab_mode = collab_mode
        if collab_mode == "conditional_lora" and float(gnn_scale) != 0:
            raise ValueError("conditional_lora requires gnn_scale=0")
        if collab_mode != "legacy" and (rel_rank_mod or rel_inject):
            raise ValueError("Disable rel_rank_mod/rel_inject for isolated experiments")
        self.inj_diag = bool(inj_diag)
        self.inj_warmup_ratio = float(gnn_injection_warmup_ratio)
        self.inj_warmup_total_steps = int(gnn_injection_warmup_total_steps)
        if self.inj_warmup_ratio > 0:
            _ws = int(self.inj_warmup_ratio * self.inj_warmup_total_steps)
            print(
                f"[A1] gnn injection warm-up ON: ratio={self.inj_warmup_ratio} "
                f"x total_steps={self.inj_warmup_total_steps} -> warmup_steps={_ws} "
                f"(gnn_scale ramps 0 -> gnn_scale, linear)",
                flush=True
            )
        # Confidence-weighted consensus teacher (CW series): the semantic
        # target becomes the MEAN item-item relation over several teacher
        # layers, weighted per-pair by cross-layer agreement
        # c_ij = exp(-Var_l S_ij / tau). Diagnostic on the champion showed
        # high-uncertainty pairs carry ~1.2x larger |S8-S_G| error, so the
        # consensus target differs from S8 exactly where supervision is
        # weakest. Teacher-only change: GNN gradient path is untouched.
        self.sem_consensus = bool(sem_consensus)
        self.sem_conf_tau = float(sem_conf_tau)
        # FX fix (2026-10-05): the semantic teacher pass runs the LLM under
        # no_grad + autocast BEFORE the main forward, inside the same outer
        # autocast region. PyTorch's autocast weight cache then holds fp16
        # casts of the layers the teacher visited (0..exit_layer-1) created
        # WITHOUT grad tracking, and the main forward reuses them — cutting
        # those layers' LoRA weights out of the autograd graph entirely
        # (probed: layers 0-7 receive EXACTLY zero grads, boundary = teacher
        # exit_layer). Disabling the cache for the teacher pass restores the
        # full 32-layer adapter training. Default off = historical behaviour.
        self.fix_autocast_cache = bool(fix_autocast_cache)
        if self.fix_autocast_cache:
            print("[FX] autocast cache fix ON: teacher pass uses cache_enabled=False "
                  "(layers below teacher exit_layer regain gradient flow)", flush=True)
        # target of the confidence-weighted loss: "mean" = layer consensus
        # (CW1, collapsed), "s8" = plain layer-8 target with confidence
        # weights only (CW2, isolates the weighting from the target shift)
        self.sem_consensus_target = str(sem_consensus_target)
        self._sem_consensus_layers = [
            int(x) for x in str(sem_consensus_layers).split(",") if x != ""
        ] or [8]
        if 8 not in self._sem_consensus_layers:
            self._sem_consensus_layers.append(8)
        self._sem_consensus_layers = sorted(set(self._sem_consensus_layers))
        if self.sem_consensus:
            print(
                f"[CW] consensus teacher ON: layers={self._sem_consensus_layers}, "
                f"conf_tau={self.sem_conf_tau}",
                flush=True
            )
        if self.sem_adapter_on:
            e = rec_config.embedding_size
            # fork_rng: the adapter init consumes an ISOLATED RNG stream so the
            # global init stream (llama_proj etc.) stays identical to the
            # no-adapter baseline — any R-series vs A difference is then purely
            # the mechanism, not init lottery
            with torch.random.fork_rng(devices=[]):
                # CPU-only seeding: torch.manual_seed would ALSO reseed every
                # CUDA device, and fork_rng(devices=[]) does not restore CUDA
                # state — that leak shifted all dropout masks in training and
                # forked the trajectory (the R0S v1 failure)
                torch.random.default_generator.manual_seed(20260930)
                self.sem_adapter = nn.Sequential(
                    nn.Linear(e, e),
                    nn.ReLU(),
                    nn.Linear(e, e),
                )
            nn.init.zeros_(self.sem_adapter[-1].weight)
            nn.init.zeros_(self.sem_adapter[-1].bias)
        # Relation-aware collaborative injection (all at the ORIGINAL single
        # injection layer, all zero-init => training starts at the baseline):
        #   rel_inject : step 2 — pair vector c_ui = enc([z_u;z_t;z_u*z_t;|z_u-z_t|])
        #                added to the user/target node rows of the injected tensor
        #   rel_gate   : step 3 — per-sample scalar gate on c_ui
        #   rel_rank_mod : step 4 — rank-wise modulation of the LoRA-A output
        #                by 2*sigmoid(MLP(c_ui)) (identity at init), replacing
        #                the additive z_G term (set gnn_scale=0 in the config)
        self.rel_inject = bool(rel_inject)
        self.rel_gate = bool(rel_gate)
        self.rel_rank_mod = bool(rel_rank_mod)
        if self.rel_inject or self.rel_rank_mod:
            e = rec_config.embedding_size
            r_dim = int(lora_config.r)
            if self.rel_inject:
                self.rel_encoder = nn.Sequential(
                    nn.Linear(4 * e, e),
                    nn.ReLU(),
                    nn.Linear(e, r_dim),
                )
                nn.init.zeros_(self.rel_encoder[-1].weight)
                nn.init.zeros_(self.rel_encoder[-1].bias)
                if self.rel_gate:
                    self.rel_gate_vec = nn.Linear(r_dim, 1)
                    nn.init.zeros_(self.rel_gate_vec.weight)
                    nn.init.zeros_(self.rel_gate_vec.bias)
            if self.rel_rank_mod:
                self.rel_rank_mlp = nn.Sequential(
                    nn.Linear(4 * e, e),
                    nn.ReLU(),
                    nn.Linear(e, r_dim),
                )
                nn.init.zeros_(self.rel_rank_mlp[-1].weight)
                nn.init.zeros_(self.rel_rank_mlp[-1].bias)

        print("running MiniGPT4Rec_v2 ...... ")
        print('Loading Rec_model')
        self.rec_model_type = rec_model
        self.rec_encoder = self.init_rec_encoder(rec_model, rec_config, rec_precision)
        if self.rec_encoder is not None and pretrained_rec != "not_have":
            self.rec_encoder.load_state_dict(torch.load(pretrained_rec, map_location="cpu"))
            print("successfully load the pretrained model......")
        if freeze_rec and self.rec_encoder is not None:
            for name, param in self.rec_encoder.named_parameters():
                param.requires_grad = False
            self.rec_encoder = self.rec_encoder.eval()
            self.rec_encoder.train = disabled_train
            logging.info("freeze rec encoder")
            print("freeze rec encoder")
        print('Loading Rec_model Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.use_lora = False
        self.gcn = None
        self.my_linears = []

        if lora_config is not None and lora_config.use_lora:
            print("Setting LoRA")
            self.use_lora = True
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model_lora = get_peft_model(self.llama_model, peft_config)

            if lora_gnn == "lightgcn":
                self.gcn = LightGCN(in_dim=256, out_dim=lora_config.r, num_layers=3, dropout=0.1)
            elif lora_gnn == "ngcf":
                self.gcn = NGCF(in_dim=256, hidden_dim=256, out_dim=lora_config.r, num_layers=3, dropout=0.1)
            elif lora_gnn == "graphsage":
                self.gcn = GraphSAGE(in_dim=256, hidden_dim=256, out_dim=lora_config.r, dropout=0.1, sample_size=25)
            elif lora_gnn == "gat":
                self.gcn = GAT(in_dim=256, hidden_dim=256, out_dim=lora_config.r, dropout=0.1, alpha=0.2)
            elif lora_gnn == "gcn":
                self.gcn = TwoLayerGCN(in_dim=256, hidden_dim=256, out_dim=lora_config.r, dropout=0.1)
            else:
                raise ValueError(
                    f"Unsupported lora_gnn type: {lora_gnn}. "
                    "Choose from ['lightgcn', 'ngcf', 'graphsage', 'gat', 'gcn']"
                )

            self.num_users = rec_config.user_num
            self.num_items = rec_config.item_num

            if self.gnn_semantic_align and lora_gnn not in ("ngcf", "lightgcn"):
                raise ValueError(
                    "gnn_semantic_align currently requires lora_gnn='ngcf' or 'lightgcn' "
                    "because semantic correction uses the 256-d penultimate GNN representation."
                )

            target_layer_str = f'model.layers.{gcn_layer}'
            print(f"Preparing to inject GCN into: {target_layer_str} with gnn_scale: {gnn_scale}")

            for name, module in self.llama_model_lora.named_modules():
                if target_layer_str in name and any(t in name for t in ['q_proj', 'v_proj']):
                    if isinstance(module, OriginalLinear):
                        new_module = MyLinear(
                            original_linear=module,
                            layer_idx=gcn_layer,
                            lora_scale=lora_scale,
                            gnn_scale=gnn_scale,
                            gcn_layer=gcn_layer,
                            collab_mode=self.collab_mode,
                            module_tag='q' if 'q_proj' in name else 'v'
                        )
                        new_module.inj_diag = self.inj_diag

                        parent_name = '.'.join(name.split('.')[:-1])
                        attr_name = name.split('.')[-1]
                        parent_module = self.llama_model_lora
                        for part in parent_name.split('.'):
                            parent_module = getattr(parent_module, part)
                        setattr(parent_module, attr_name, new_module)
                        self.my_linears.append(new_module)

            print(
                f"Setting LoRA, GCN, and MyLinear for layer {gcn_layer} Done. "
                f"Replaced {len(self.my_linears)} modules."
            )

            # Budget-neutral per-layer LoRA rank redistribution (RC series).
            # lora_rank_groups: list of [start_layer, end_layer, rank]
            # (inclusive). The injection layer is NEVER re-ranked — its
            # bottleneck IS the GraphLoRA collaborative projection. Modules
            # in a group whose rank equals the base rank are not touched, so
            # a fully uniform grouping leaves the path bitwise identical to
            # plain LoRA. LoRA scaling follows alpha_l/r_l = base ratio
            # (hardcoded alpha=16, r=8 above => ratio 2) so capacity, not
            # update scale, is the only changed factor. Re-init is wrapped in
            # fork_rng + CPU-only seeding so the global init stream
            # (llama_proj etc.) stays bit-identical (RNG-leak lesson 10-02).
            if lora_rank_groups:
                _base_rank, _base_alpha = 8, 16
                _ratio = _base_alpha / _base_rank
                layer_rank = {}
                for _g in lora_rank_groups:
                    for _l in range(int(_g[0]), int(_g[1]) + 1):
                        layer_rank[_l] = int(_g[2])
                _total = sum(layer_rank.get(l, _base_rank) for l in range(32))
                print(
                    f"[RC] rank redistribution: groups={lora_rank_groups} "
                    f"total_rank={_total} (baseline {32 * _base_rank}), "
                    f"r_{gcn_layer} fixed at {_base_rank}",
                    flush=True
                )
                if _total != 32 * _base_rank:
                    print(f"[RC] WARNING: total rank budget is NOT budget-neutral!", flush=True)
                _n_rep = 0
                with torch.random.fork_rng(devices=[]):
                    # CPU-only seeding: global CPU stream restored on exit, so
                    # any later init (llama_proj) is untouched
                    torch.random.default_generator.manual_seed(20261004)
                    for name, module in self.llama_model_lora.named_modules():
                        _m = re.match(r".*model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)$", name)
                        if _m is None or not isinstance(module, OriginalLinear):
                            continue
                        _li = int(_m.group(1))
                        _new_r = layer_rank.get(_li, _base_rank)
                        if _new_r == _base_rank or _li == gcn_layer:
                            continue
                        _dev = module.lora_A["default"].weight.device
                        _dt = module.lora_A["default"].weight.dtype
                        _A = nn.Linear(module.in_features, _new_r, bias=False).to(device=_dev, dtype=_dt)
                        _B = nn.Linear(_new_r, module.out_features, bias=False).to(device=_dev, dtype=_dt)
                        nn.init.kaiming_uniform_(_A.weight, a=5 ** 0.5)
                        nn.init.zeros_(_B.weight)  # B=0 => dW=0 at init
                        module.lora_A["default"] = _A
                        module.lora_B["default"] = _B
                        module.scaling["default"] = _ratio * _new_r
                        module.r["default"] = _new_r
                        _n_rep += 1
                print(f"[RC] re-ranked {_n_rep} LoRA modules (A kaiming, B zero-init)", flush=True)

            if self.gnn_semantic_align:
                print(
                    f"GNN semantic correction ON: teacher_layer={self._gnn_semantic_layer}, "
                    f"weight={self.gnn_semantic_weight}, "
                    f"rec_loss_updates_gnn={self.train_gnn_with_rec_loss}"
                )

        if freeze_lora and self.use_lora:
            print("freeze LoRA...")
            for name, param in self.llama_model_lora.named_parameters():
                param.requires_grad = False

        if self.rec_encoder is not None and 'prompt' not in rec_model:
            self.llama_proj = nn.Sequential(
                nn.Linear(
                    self.rec_encoder.config.embedding_size,
                    self.rec_encoder.config.embedding_size * int(proj_mid)
                ),
                nn.ReLU(),
                nn.Linear(
                    self.rec_encoder.config.embedding_size * int(proj_mid),
                    self.llama_model.config.hidden_size * self.proj_token_num
                ),
            )
        elif self.rec_encoder is not None and rec_model == "personlized_prompt":
            self.llama_proj = nn.Linear(
                rec_config.item_num + rec_config.user_num,
                self.llama_model.config.hidden_size * self.proj_token_num,
                bias=False
            )
        elif self.rec_encoder is not None and rec_model == "soft_prompt":
            self.llama_proj = nn.Linear(
                2,
                self.llama_model.config.hidden_size * self.proj_token_num,
                bias=False
            )
        else:
            self.llama_proj = None

        if freeze_proj and self.llama_proj is not None:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            self.llama_proj = self.llama_proj.eval()
            self.llama_proj.train = disabled_train
            logging.info("!!!! freeze llama_proj...")

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.has_print_prompt = False

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt List: \n{}'.format(self.prompt_list))
            self.has_pri_decode = False
            self.prompt_list_p = None
        else:
            self.prompt_list = []
            self.prompt_list_p = None

        self._load_adjacency_matrix(mode='train')

    def _load_adjacency_matrix(self, mode='train'):
        try:
            if self.dataset == "book":
                # print("book adj")
                adj_matrix_paths = {
                    'train': "/mnt/disk2/wgj/wgj_collm/collm-datasets/amazon_book/book/book_graph_train.npz",
                    'valid': "/mnt/disk2/wgj/wgj_collm/collm-datasets/amazon_book/book/book_graph_valid.npz",
                    'test': "/mnt/disk2/wgj/wgj_collm/collm-datasets/amazon_book/book/book_graph_test.npz"
                }

            elif self.dataset == "movie":
                adj_matrix_paths = {
                    'train': "./dataset/ml-1m/movie_graph_train.npz",
                    'valid': "./dataset/ml-1m/movie_graph_valid.npz",
                    'test': "./dataset/ml-1m/movie_graph_test.npz"
                }

            adj_matrix_path = adj_matrix_paths.get(mode, adj_matrix_paths['train'])
            # print(f"Loading adjacency matrix for {mode} from: {adj_matrix_path}")

            # 加载 .npz 文件
            norm_adj = sp.load_npz(adj_matrix_path)
            # print(f"Loaded adjacency matrix shape: {norm_adj.shape}, non-zero elements: {norm_adj.nnz}")

            # 转换为 COO 格式并存储
            self.adj_matrix_coo = norm_adj.tocoo()
            self.adj_matrix_shape = norm_adj.shape
            self.adj_matrix_csr = norm_adj.tocsr()
            # print(f"Stored COO matrix, shape: {self.adj_matrix_shape}")
        except Exception as e:
            print(f"Failed to load adjacency matrix: {e}")
            raise

    def get_subgraph(self, node_ids):
        DEFAULT_HOP = 1
        MAX_NODES = None
        hop = int(DEFAULT_HOP)
        if hop < 0:
            hop = 0
        if not hasattr(self, "adj_matrix_csr") or self.adj_matrix_csr is None:
            self.adj_matrix_csr = self.adj_matrix_coo.tocsr()
        csr = self.adj_matrix_csr


        node_ids_np = node_ids.unique().detach().cpu().numpy().astype(np.int64)
        node_ids_np = node_ids_np[(node_ids_np >= 0) & (node_ids_np < csr.shape[0])]
        if node_ids_np.size == 0:
            empty = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=self.device),
                torch.zeros((0,), dtype=torch.float32, device=self.device),
                (0, 0)
            ).coalesce()
            return empty, torch.empty((0,), dtype=torch.long, device=self.device)

        current = set(node_ids_np.tolist())
        frontier = set(current)

        # --- BFS 扩展 hop 层 ---
        for _ in range(hop):
            if not frontier:
                break

            new_neighbors = set()
            for u in frontier:
                start, end = csr.indptr[u], csr.indptr[u + 1]
                if end > start:
                    new_neighbors.update(csr.indices[start:end].tolist())

            new_neighbors.difference_update(current)
            if not new_neighbors:
                break


            if MAX_NODES is not None and (len(current) + len(new_neighbors) > MAX_NODES):
                remaining = MAX_NODES - len(current)
                if remaining <= 0:
                    break
                new_list = list(new_neighbors)
                chosen = np.random.choice(new_list, size=remaining, replace=False).tolist()
                new_neighbors = set(chosen)

            current.update(new_neighbors)
            frontier = new_neighbors

        sub_nodes = np.array(sorted(current), dtype=np.int64)
        sub_sp = csr[sub_nodes][:, sub_nodes].tocoo()

        indices = torch.tensor(
            np.vstack([sub_sp.row, sub_sp.col]),
            dtype=torch.long,
            device=self.device
        )
        values = torch.tensor(sub_sp.data, dtype=torch.float32, device=self.device)
        shape = (len(sub_nodes), len(sub_nodes))

        sub_adj_matrix = torch.sparse_coo_tensor(indices, values, size=shape).coalesce()
        sub_node_ids = torch.tensor(sub_nodes, dtype=torch.long, device=self.device)
        return sub_adj_matrix, sub_node_ids

    def compute_feedback_gcn(self, samples, samples_encode, replaced_idx,
                             inputs_embeds, attention_mask):
        """
        Compute the GraphLoRA GNN representation.

        When gnn_semantic_align=True during training, this function additionally:
          1) runs an early LLM teacher pass up to a fixed layer;
          2) extracts detached semantic hidden states at ID-token positions;
          3) returns the GNN penultimate representation for semantic-relation alignment.

        The teacher branch is stop-gradient. The semantic loss therefore updates the
        GNN side only.
        """
        node_ids = samples_encode['node_ids'].to(self.device)
        sub_adj_matrix, sub_node_ids = self.get_subgraph(node_ids)
        node_map = {old_id.item(): new_id for new_id, old_id in enumerate(sub_node_ids)}

        batch_mask = replaced_idx[:, 0] < samples['UserID'].shape[0]
        batch_indices = replaced_idx[batch_mask, 0]
        token_positions = replaced_idx[batch_mask, 1]
        batch_node_ids = node_ids[batch_mask]
        local_node_ids = torch.tensor(
            [node_map.get(nid.item(), -1) for nid in batch_node_ids],
            device=self.device
        )

        mf_embeddings = torch.cat(
            [self.rec_encoder.user_embedding.weight,
             self.rec_encoder.item_embedding.weight],
            dim=0
        )

        h_semantic = None
        need_semantic = self.training and self.gnn_semantic_align
        need_pref = self.training and self.pref_distill
        need_gnn_view = need_semantic or need_pref \
            or self.rel_inject or self.rel_rank_mod

        if need_semantic:
            # LLM is the semantic teacher for this auxiliary objective.
            # no_grad is intentional: L_sem must not update LLM/LoRA.
            with torch.no_grad():
                if self.fix_autocast_cache:
                    # cache_enabled=False: the no-grad fp16 weight casts made
                    # here must NOT enter the autocast cache, otherwise the
                    # main forward reuses them and layers 0..exit_layer-1
                    # lose their autograd connection (see [FX] note above)
                    with torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=False):
                        _exit = max(self._sem_consensus_layers) \
                            if self.sem_consensus else self._gnn_semantic_layer
                        teacher_out = self.llama_model_lora(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            return_dict=True,
                            use_cache=False,
                            exit_layer=_exit,
                            output_hidden_states=True,
                        )
                else:
                    with self.maybe_autocast():
                        _exit = max(self._sem_consensus_layers) \
                            if self.sem_consensus else self._gnn_semantic_layer
                        teacher_out = self.llama_model_lora(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            return_dict=True,
                            use_cache=False,
                            exit_layer=_exit,
                            output_hidden_states=True,
                        )
                teacher_hs = teacher_out.hidden_states
                semantic_layer = min(self._gnn_semantic_layer, len(teacher_hs) - 1)
                h_semantic = teacher_hs[semantic_layer][
                    batch_indices, token_positions
                ].float().detach()
                if self.sem_consensus:
                    # [n, L, d] stack at the configured teacher layers; the
                    # layer-8 slice stays as h_semantic for the R-series corr
                    # loss, the consensus loss reads the whole stack
                    _L = min(len(teacher_hs) - 1, max(self._sem_consensus_layers))
                    h_semantic_stack = torch.stack(
                        [
                            teacher_hs[min(l, _L)][batch_indices, token_positions].float()
                            for l in self._sem_consensus_layers
                        ],
                        dim=1,
                    ).detach()

        self.gcn.eval()
        with torch.cuda.amp.autocast(enabled=False):
            sub_adj_matrix = sub_adj_matrix.to(dtype=torch.float32, device=self.device)
            node_features = mf_embeddings[sub_node_ids].to(dtype=torch.float32, device=self.device)

            pre_last = None
            if getattr(self, 'sem_adapter', False) and getattr(self, 'sem_adapter_in_path', False):
                # Anchor-Preserving Semantic Refinement (user design 09-30):
                # h* = h + alpha * Adapter(normalize(h)) applied right before
                # the last NGCF layer. Two branches:
                #   live  -> z_G* feeds the recommendation path (rec grads
                #            reach trunk + adapter per the rec->GNN gate)
                #   corr  -> S_G* for L_corr; when sem_trunk_grad is False
                #            the trunk is detached so L_corr trains the
                #            adapter only
                sem_alpha = self.sem_alpha
                sem_adapter = self.sem_adapter
                sem_trunk_grad = self.sem_trunk_grad

                def pre_last(h_raw, penult_uncorr):
                    hn = F.normalize(h_raw, dim=-1).float()
                    delta = sem_adapter(hn)                      # zero-init => 0
                    h_star_live = h_raw + sem_alpha * delta.to(h_raw.dtype)
                    if sem_trunk_grad:
                        corr = F.normalize(h_star_live.float(), dim=-1)
                    else:
                        hn_d = hn.detach()
                        corr = F.normalize(
                            (hn_d + sem_alpha * sem_adapter(hn_d)).float(), dim=-1)
                    pack = {"star": corr, "anchor": penult_uncorr.detach()}
                    return h_star_live, pack

            if need_gnn_view:
                if isinstance(self.gcn, NGCF):
                    gcn_output, gcn_penult, gcn_pack = self.gcn(
                        sub_adj_matrix,
                        node_features,
                        sub_node_ids,
                        return_penult=True,
                        pre_last=pre_last,
                    )
                else:
                    gcn_output, gcn_penult = self.gcn(
                        sub_adj_matrix,
                        node_features,
                        sub_node_ids,
                        return_penult=True
                    )
                    gcn_pack = None
            else:
                gcn_output = self.gcn(
                    sub_adj_matrix,
                    node_features,
                    sub_node_ids
                )
                gcn_penult = None
                gcn_pack = None

        ret = {
            "gcn_output": gcn_output,
            "batch_indices": batch_indices,
            "token_positions": token_positions,
            "local_node_ids": local_node_ids,
            "step": int(samples.get('epoch', 0)) * int(samples.get('num_iters_per_epoch', 0))
                    + int(samples.get('iters', 0)),
        }

        if need_gnn_view:
            valid = local_node_ids >= 0
            token_gnn = torch.zeros(
                (local_node_ids.shape[0], gcn_penult.shape[1]),
                dtype=gcn_penult.dtype,
                device=self.device
            )
            token_gnn[valid] = gcn_penult[local_node_ids[valid]]

            ret["token_gnn"] = token_gnn
            ret["token_node_ids"] = batch_node_ids

            if gcn_pack is not None:
                token_gnn_star = torch.zeros_like(token_gnn)
                token_gnn_star[valid] = gcn_pack["star"][local_node_ids[valid]]
                token_gnn_anchor = torch.zeros_like(token_gnn)
                token_gnn_anchor[valid] = gcn_pack["anchor"][local_node_ids[valid]]
                ret["token_gnn_star"] = token_gnn_star
                ret["token_gnn_anchor"] = token_gnn_anchor
            elif self.sem_adapter_on:
                # Stage 1 (adapter not in the rec path): the corrected
                # representation exists only in the loss, computed from the
                # detached penult so L_corr trains the adapter exclusively
                tg_d = token_gnn.detach().float()
                star_raw = tg_d + self.sem_alpha * self.sem_adapter(tg_d)
                ret["token_gnn_star"] = F.normalize(star_raw, dim=-1).to(token_gnn.dtype)
                ret["token_gnn_anchor"] = token_gnn.detach()

        if self.rel_inject or self.rel_rank_mod:
            # pair features: per batch row, the USER token and TARGET token of
            # token_gnn (256-d penult rows) form the relation input
            user_mask = batch_node_ids < self.num_users
            u_tok, t_tok = [], []
            for r in batch_indices.unique():
                rows_r = (batch_indices == r).nonzero(as_tuple=True)[0]
                u_rows = rows_r[user_mask[rows_r]]
                i_rows = rows_r[~user_mask[rows_r]]
                if u_rows.numel() and i_rows.numel():
                    u_tok.append(u_rows[0].item())
                    t_tok.append(i_rows[-1].item())
            if u_tok:
                u_tok = torch.tensor(u_tok, device=self.device)
                t_tok = torch.tensor(t_tok, device=self.device)
                z_u = token_gnn[u_tok].float() if need_gnn_view else None
                z_t = token_gnn[t_tok].float() if need_gnn_view else None
                if z_u is not None:
                    feats = torch.cat([z_u, z_t, z_u * z_t, (z_u - z_t).abs()], dim=-1)
                    u_node = local_node_ids[u_tok]
                    t_node = local_node_ids[t_tok]
                    if self.rel_inject:
                        c = self.rel_encoder(feats)
                        if self.rel_gate:
                            c = c * torch.sigmoid(self.rel_gate_vec(c))
                        ret["pair_add"] = {"c": c, "u_node": u_node, "t_node": t_node}
                    if self.rel_rank_mod:
                        ret["pair_rankgate"] = {
                            "g_raw": self.rel_rank_mlp(feats),
                            "rows": batch_indices.unique(),
                        }

        if need_semantic:
            ret["h_semantic"] = h_semantic
            if self.sem_consensus:
                ret["h_semantic_stack"] = h_semantic_stack

        return ret

    def _compute_gnn_semantic_loss(self, fb):
        """
        LLM -> GNN semantic correction using item-item relation alignment.

        S_L(i,j) = cos(h_i, h_j) from detached early-layer LLM states.
        S_G(i,j) = cos(z_i, z_j) from the trainable GNN penultimate states.

        L_sem = mean_{i<j, item_i!=item_j} (S_G(i,j) - sg(S_L(i,j)))^2

        Only the GNN side receives gradients from L_sem.
        """
        h_semantic = fb["h_semantic"]
        token_gnn = fb["token_gnn"]
        token_node_ids = fb["token_node_ids"]
        local_node_ids = fb["local_node_ids"]

        valid_items = (token_node_ids >= self.num_users) & (local_node_ids >= 0)
        item_rows = valid_items.nonzero(as_tuple=True)[0]

        if item_rows.numel() < 2:
            return token_gnn.sum() * 0.0

        h_items = F.normalize(h_semantic[item_rows].float(), dim=-1)
        z_items = F.normalize(token_gnn[item_rows].float(), dim=-1)
        item_ids = token_node_ids[item_rows]

        sim_llm = h_items @ h_items.t()
        sim_gnn = z_items @ z_items.t()

        pair_mask = torch.triu(
            torch.ones_like(sim_gnn, dtype=torch.bool),
            diagonal=1
        )
        # Avoid trivial same-ID pairs if the same item appears more than once.
        pair_mask = pair_mask & (item_ids.unsqueeze(1) != item_ids.unsqueeze(0))

        if not bool(pair_mask.any()):
            return token_gnn.sum() * 0.0

        if "h_semantic_stack" in fb:
            # CW consensus/confidence teacher: per-pair confidence
            # c = exp(-Var_l S_l / tau). Target is the layer mean ("mean",
            # CW1) or the plain layer-8 relation ("s8", CW2 ablation).
            hs_stack = F.normalize(fb["h_semantic_stack"][item_rows].float(), dim=-1)
            s_layers = torch.stack(
                [hs_stack[:, l] @ hs_stack[:, l].transpose(0, 1)
                 for l in range(hs_stack.shape[1])],
                dim=0,
            )
            if self.sem_consensus_target == "s8":
                target = sim_llm.detach()
            else:
                target = s_layers.mean(dim=0).detach()
            conf = torch.exp(-s_layers.var(dim=0, unbiased=False) / self.sem_conf_tau)
            resid = (sim_gnn - target) ** 2
            return (resid[pair_mask] * conf[pair_mask]).sum() / conf[pair_mask].sum().clamp_min(1e-8)

        target = sim_llm.detach()
        return ((sim_gnn - target) ** 2)[pair_mask].mean()

    def _compute_corr_loss(self, fb):
        """Anchor-preserving semantic correction loss (user design 09-30).

        target = (1-rho)*sg(S_G_anchor) + rho*sg(S_L)
        L_corr = mean over item pairs of (S_G* - target)^2

        S_G* comes from the adapter-corrected penult (token_gnn_star); the
        anchor is the uncorrected penult relations (token_gnn_anchor), both
        detached where the stage requires it.
        """
        tgs = fb["token_gnn_star"]
        tga = fb["token_gnn_anchor"]
        token_node_ids = fb["token_node_ids"]
        local_node_ids = fb["local_node_ids"]

        valid_items = (token_node_ids >= self.num_users) & (local_node_ids >= 0)
        item_rows = valid_items.nonzero(as_tuple=True)[0]
        if item_rows.numel() < 2:
            return tgs.sum() * 0.0

        hs_items = F.normalize(fb["h_semantic"][item_rows].float(), dim=-1)
        star_items = F.normalize(tgs[item_rows].float(), dim=-1)
        anchor_items = F.normalize(tga[item_rows].float(), dim=-1)
        item_ids = token_node_ids[item_rows]

        S_L = (hs_items @ hs_items.t()).detach()
        S_anchor = (anchor_items @ anchor_items.t()).detach()
        S_star = star_items @ star_items.t()

        pair_mask = torch.triu(torch.ones_like(S_star, dtype=torch.bool), diagonal=1)
        pair_mask = pair_mask & (item_ids.unsqueeze(1) != item_ids.unsqueeze(0))
        if not bool(pair_mask.any()):
            return tgs.sum() * 0.0

        rho_t = self.sem_rho
        if self.sem_decay:
            t = fb.get("step", 0)
            if t >= self.sem_decay_t2:
                rho_t = 0.0
            elif t > self.sem_decay_t1:
                rho_t = self.sem_rho * (self.sem_decay_t2 - t) / float(self.sem_decay_t2 - self.sem_decay_t1)
        target = (1.0 - rho_t) * S_anchor + rho_t * S_L
        return ((S_star - target) ** 2)[pair_mask].mean()

    def _row_gnn_scores(self, fb):
        """Per-batch-row collaborative score s_G(r) = kappa * cos(z_user, z_target).

        Each batch row contributes one user token and one target item token
        (the last item-position row), so the GNN's opinion on that row's
        candidate is a single scalar. Requires fb to carry token_gnn.
        """
        token_gnn = fb["token_gnn"]
        token_node_ids = fb["token_node_ids"]
        batch_indices = fb["batch_indices"]
        user_mask = token_node_ids < self.num_users
        tgt_rows = []
        for r in batch_indices.unique():
            rows_r = (batch_indices == r).nonzero(as_tuple=True)[0]
            item_rows_r = rows_r[~user_mask[rows_r]]
            tgt_rows.append(item_rows_r[-1] if item_rows_r.numel() > 0 else rows_r[0])
        tgt_rows = torch.stack(tgt_rows)
        user_rows = user_mask.nonzero(as_tuple=True)[0]
        kap = self.gnn_conf_kappa if hasattr(self, "gnn_conf_kappa") else 5.0
        return kap * F.cosine_similarity(token_gnn[user_rows], token_gnn[tgt_rows], dim=-1)

    def _gnn_conf_loss(self, fb, samples):
        """GNN-side recommendation loss: BCE on kappa*cos(z_u, z_target).

        Gives the GNN (and the calibration scalar kappa) a direct task signal
        that does not depend on the LoRA injection path.
        """
        s_g_row = self._row_gnn_scores(fb)
        labels = samples['label'].float().to(s_g_row.dtype)
        return F.binary_cross_entropy_with_logits(s_g_row.float(), labels)

    def _pref_distill_loss(self, fb, logits, samples):
        """Preference-level G->L distillation over candidate groups.

        For each candidate group (one user/history prompt with K candidate
        targets), the GNN's preference distribution
            P_G = softmax(kappa*cos(z_u,z_c)/tau_G)      (teacher, stop-grad)
        is distilled into the LLM's Yes-probability distribution
            P_L = softmax(yes_logit/tau_L)               (student)
        weighted by the GNN's confidence.
        """
        s_g_row = self._row_gnn_scores(fb)
        q_g_row = torch.sigmoid(s_g_row.float()).detach()
        groups = samples["cand_group"]
        loss_sum = logits.sum() * 0.0
        w_sum = torch.zeros((), device=logits.device)
        with torch.cuda.amp.autocast(enabled=False):
            for g in groups.unique():
                rows_g = (groups == g).nonzero(as_tuple=True)[0]
                if rows_g.numel() < 2:
                    continue
                p_l = F.log_softmax(logits[rows_g].float() / self.pref_tau_l, dim=-1)
                p_g = F.softmax(s_g_row[rows_g].float() / self.pref_tau_g, dim=-1).detach()
                w = q_g_row[rows_g].mean()
                kl = (p_g * (torch.log(p_g.clamp_min(1e-12)) - p_l)).sum()
                loss_sum = loss_sum + w * kl
                w_sum = w_sum + w
        return loss_sum / w_sum.clamp_min(1e-6)


    def to_be_trained(self):
        if self.use_lora:
            return True
        id_terms = ["<UserID>", "<ItemIDList>", "<TargetItemID>", "<DCNFeature>"]
        for prompt in self.prompt_list:
            for id_term in id_terms:
                if id_term in prompt:
                    return True
        return False

    def set_mode(self, mode):
        self.run_mode_ = mode

    def rec_to_cpu(self):
        self.rec_encoder.to("cpu")
        self.rec_encoder.float()

    def set_answer_type(self, mode):
        if mode == 'v1':
            self.pos_ans = ["former"]
            self.neg_ans = ["latter"]
        elif mode == 'v2':
            self.pos_ans = ['Yes']
            self.neg_ans = ['No']
            pos_ans_id = self.llama_tokenizer(self.pos_ans[0], add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(self.neg_ans[0], add_special_tokens=False).input_ids[0]
            print("answer token ids: pos:", pos_ans_id, "neg ids:", neg_ans_id)
        else:
            raise NotImplementedError("not implement this types of answers")

    def print_prompt(self):
        print('Prompt Pos Example \n{} {} or {}'.format(random.choice(self.prompt_list), self.pos_ans[0],
                                                        self.neg_ans[0]))

    def encode_recdata_v1(self, sample):
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        with self.maybe_autocast():
            all_user_embeds, all_items_embeds = self.rec_encoder.computer()
            user_embeds = self.rec_encoder.user_encoder(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
            targetItem_embed = self.rec_encoder.item_encoder(sample['PairItemIDs'], all_items=all_items_embeds)
            user_embeds_llama = self.llama_proj(user_embeds)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed)
        sample_embeds_llama = {
            'User_emb': user_embeds_llama,
            'PairItem_emb': targetItem_embeds_llama,
        }
        sample_atts_llama = None
        return sample_embeds_llama, sample_atts_llama

    def encode_recdata_v2(self, sample, ids_order=None):
        if self.rec_encoder is None:
            return None, None
        device = sample['UserID'].device
        if self.low_resource:
            self.rec_to_cpu()
            for key in sample:
                sample[key] = sample[key].to('cpu')
        with self.maybe_autocast():
            batch_size = sample['UserID'].shape[0]
            hidden_size = self.llama_model.config.hidden_size
            all_user_embeds, all_item_embeds = self.rec_encoder.computer()
            if self.rec_model_type == "sasrec":
                user_embeds = self.rec_encoder.seq_encoder(sample['sas_seq']).unsqueeze(-2)
            elif self.rec_model_type == "DCN" or self.rec_model_type == "DIN":
                user_embeds = self.rec_encoder.all_encode(sample['UserID'], sample['TargetItemID'],
                                                          sample['sas_seq'][:, -10:]).unsqueeze(-2)
            else:
                user_embeds = self.rec_encoder.user_encoder(sample['UserID'], all_users=all_user_embeds).unsqueeze(-2)
            targetItem_embed = self.rec_encoder.item_encoder(sample['TargetItemID'],
                                                             all_items=all_item_embeds).unsqueeze(-2)
            user_embeds_llama = self.llama_proj(user_embeds).reshape(batch_size, -1, self.proj_token_num, hidden_size)
            targetItem_embeds_llama = self.llama_proj(targetItem_embed).reshape(batch_size, -1, self.proj_token_num,
                                                                                hidden_size)
            num_users = self.rec_encoder.config.user_num
            node_ids = []
            if 'InteractedItemIDs_pad' in sample.keys() and len(ids_order) == 3:
                interactedItem_embeds = self.rec_encoder.item_encoder(sample['InteractedItemIDs_pad'],
                                                                      all_items=all_item_embeds)
                interactedItem_embeds_llama = self.llama_proj(interactedItem_embeds).reshape(batch_size, -1,
                                                                                             self.proj_token_num,
                                                                                             hidden_size)
                merged_embeds = [user_embeds_llama, interactedItem_embeds_llama, targetItem_embeds_llama]
                merged_embeds = [merged_embeds[k] for k in ids_order]
                merged_embeds = torch.cat(merged_embeds, dim=1)
                idx_flag = torch.ones_like(sample['InteractedItemIDs_pad'])
                idx_flag = torch.where(sample['InteractedItemIDs_pad'] == self.rec_encoder.padding_index, 0, idx_flag)
                idx_flag = [torch.ones([idx_flag.shape[0], 1]).to(idx_flag.device), idx_flag,
                            torch.ones([idx_flag.shape[0], 1]).to(idx_flag.device)]
                idx_flag = [idx_flag[k] for k in ids_order]
                idx_flag = torch.cat(idx_flag, dim=1).to(device)
                idx_nopad = torch.nonzero(idx_flag)
                # segment boundaries in the ids_order-permuted layout:
                # ids_order[i] is the original segment (0=user, 1=items, 2=target)
                # placed at slot i, so position checks must follow the permutation
                # instead of assuming the user segment comes first
                seg_size = {0: 1, 1: int(sample['InteractedItemIDs_pad'].shape[1]), 2: 1}
                seg_range, start = {}, 0
                for slot, orig_seg in enumerate(ids_order):
                    orig_seg = int(orig_seg)
                    seg_range[orig_seg] = (start, start + seg_size[orig_seg])
                    start += seg_size[orig_seg]
                u_s, u_e = seg_range[0]
                i_s, i_e = seg_range[1]
                for b in range(batch_size):
                    batch_mask = idx_nopad[:, 0] == b
                    batch_idx_nopad = idx_nopad[batch_mask]
                    for _, pos in batch_idx_nopad:
                        if u_s <= pos < u_e:  # UserID
                            node_ids.append(sample['UserID'][b].item())
                        elif i_s <= pos < i_e:  # InteractedItemIDs
                            item_pos = pos - i_s
                            item_id = sample['InteractedItemIDs_pad'][b, item_pos].item()
                            if item_id != self.rec_encoder.padding_index:
                                node_ids.append(item_id + num_users)
                        else:  # TargetItemID
                            node_ids.append(sample['TargetItemID'][b].item() + num_users)
                node_ids = torch.tensor(node_ids, device=device, dtype=torch.long)
                mf_user_embeds = self.rec_encoder.user_embedding.weight
                mf_item_embeds = self.rec_encoder.item_embedding.weight
                mf_embeddings = torch.cat([mf_user_embeds, mf_item_embeds], dim=0)
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'InteractedItems_embs': interactedItem_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'merged_embs': merged_embeds[idx_nopad[:, 0], idx_nopad[:, 1]].reshape(-1, hidden_size),
                    'node_ids': node_ids,
                    'mf_embeddings': mf_embeddings,
                }
            else:
                node_ids = torch.cat([
                    sample['UserID'],
                    sample['TargetItemID'] + num_users
                ], dim=0)
                mf_user_embeds = self.rec_encoder.user_embedding.weight
                mf_item_embeds = self.rec_encoder.item_embedding.weight
                mf_embeddings = torch.cat([mf_user_embeds, mf_item_embeds], dim=0)
                sample_embeds_llama = {
                    'User_emb': user_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'TargetItem_emb': targetItem_embeds_llama.reshape(batch_size, -1, hidden_size),
                    'InteractedItems_embs': None,
                    'merged_embs': None,
                    'node_ids': node_ids,
                    'mf_embeddings': mf_embeddings,
                }
        sample_atts_llama = None
        return sample_embeds_llama, sample_atts_llama

    def recprompt_wrap_v1(self, samples, ori_samples, atts_sample, prompt):
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemID>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token
            prompt = bos + prompt
            prompt = prompt.replace("<UserID>", unk_)
            prompt = prompt.replace("<ItemID>", unk_)
            prompt_list = []
            for k in range(batch_size):
                prompt_ = prompt + ""
                prompt_list.append(prompt_)
            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
                prompt_list,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(samples['User_emb'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)
            prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            if "<UserID>" in prompt_ori and "<ItemID>" in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = torch.cat(
                    [samples['User_emb'], samples['PairItem_emb']], dim=-2
                ).reshape(-1, samples['User_emb'].shape[-1])
            else:
                raise RuntimeError("the pretraining just support one type prompt")
            return prompt_embeds, prompts_tokens.attention_mask

    def recprompt_wrap_v2(self, samples, ori_samples, atts_sample, prompt):
        if prompt:
            prompt_ori = prompt
            split_symbol = ["<UserID>", "<ItemIDList>", "<ItemTitleList>", "<TargetItemID>", "<TargetItemTitle>",
                            "<DCNFeature>"]
            batch_size = ori_samples['UserID'].shape[0]
            bos = "<s>"
            unk_ = self.llama_tokenizer.unk_token
            unk_ = ".".join([unk_] * self.proj_token_num)
            prompt = bos + prompt
            prompt = prompt.replace("<UserID>", unk_)
            prompt = prompt.replace("<TargetItemID>", unk_)
            prompt = prompt.replace("<DCNFeature>", unk_)
            prompt_list = []
            for k in range(batch_size):
                prompt_ = prompt + ""
                if 'InteractedNum' in ori_samples.keys():
                    prompt_ = prompt_.replace('<ItemIDList>', ', '.join([unk_] * ori_samples['InteractedNum'][k]))
                    prompt_ = prompt_.replace("<ItemTitleList>", ori_samples['InteractedItemTitles'][k])
                prompt_ = prompt_.replace("<TargetItemTitle>", ori_samples['TargetItemTitle'][k])
                prompt_list.append(prompt_)
            if not self.has_print_prompt:
                print("prompt example:", random.choice(prompt_list))
                self.has_print_prompt = True
            self.llama_tokenizer.padding_side = "left"
            prompts_tokens = self.llama_tokenizer(
                prompt_list,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(ori_samples['UserID'].device)
            unk_token_id = self.llama_tokenizer.unk_token_id
            if not self.has_pri_decode:
                print("#######prompt decoded example:",
                      ' '.join(self.llama_tokenizer.batch_decode(prompts_tokens.input_ids[0])))
                self.has_pri_decode = True
            replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)
            node_ids = samples['node_ids']

            num_found = replaced_idx.shape[0]
            num_target = node_ids.shape[0]

            if num_found != num_target:
                print(
                    f"[Warning-Fix] Shape mismatch detected! Found {num_found} tokens but expected {num_target} nodes.")
                print(f"Sample UserID: {samples.get('UserID', 'Unknown')}")
                min_len = min(num_found, num_target)
                replaced_idx = replaced_idx[:min_len]
                node_ids = node_ids[:min_len]
                # keep all three aligned (merged_embs rows map 1:1 to node_ids);
                # truncating only replaced_idx used to crash the index assignment
                if samples.get('merged_embs') is not None and samples['merged_embs'].shape[0] > min_len:
                    samples['merged_embs'] = samples['merged_embs'][:min_len]
            replaced_idx_with_node = torch.cat([
                replaced_idx,
                node_ids.unsqueeze(-1)
            ], dim=-1)


            prompt_embeds = self.llama_model.model.embed_tokens(prompts_tokens.input_ids)
            if "<UserID>" in prompt_ori and "<ItemIDList>" in prompt_ori and "<TargetItemID>" in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = samples['merged_embs']
            elif "<UserID>" in prompt_ori and "<TargetItemID>" in prompt_ori and "<ItemIDList>" not in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = torch.cat(
                    [samples['User_emb'], samples['TargetItem_emb']], dim=-2
                ).reshape(-1, samples['User_emb'].shape[-1])
            elif "<DCNFeature>" in prompt_ori:
                prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = samples['User_emb'].reshape(-1, samples[
                    'User_emb'].shape[-1])
            else:
                pass
            return prompt_embeds, prompts_tokens.attention_mask, replaced_idx

    def prompt_with_p(self, p):
        if self.prompt_list_p is None:
            prompt_list_p = []
            for k in range(len(p)):
                prompt_list_p.extend([self.prompt_list[k]] * p[k])
            self.prompt_list_p = prompt_list_p
            return self.prompt_list_p
        else:
            return self.prompt_list_p

    def forward(self, samples):
        # 动态加载邻接矩阵基于样本的模式
        mode = 'train'
        if 'is_valid' in samples and samples['is_valid']:
            mode = 'valid'
        elif 'is_test' in samples and samples['is_test']:
            mode = 'test'
        self._load_adjacency_matrix(mode=mode)
        # print(f"[forward] Loaded adjacency matrix for mode: {mode}")

        if self.run_mode_ == 'v1':
            return self.forward_v1(samples)
        elif self.run_mode_ == 'v2':
            return self.forward_v2(samples)
        else:
            raise NotImplementedError("None-template version has not been implemented...")

    def prompt_based_encode_v2(self, prompt, samples):
        """
        Encode prompt and samples into embeddings and attention masks for LLaMA,
        and prepare node_ids and replaced_idx for GCN.

        Args:
            prompt (str): The prompt template with placeholders.
            samples (dict): Contains 'UserID', 'TargetItemID', 'label', and optionally 'InteractedItemIDs_pad', 'InteractedNum', 'InteractedItemTitles', 'TargetItemTitle'.

        Returns:
            sample_embeds (torch.Tensor): Input embeddings for LLaMA.
            atts_samples (torch.Tensor): Attention mask for LLaMA.
            samples_encode (dict): Contains 'node_ids' and 'replaced_idx' for GCN.
        """
        id_orders = get_ids_order(prompt)
        samples_encode, atts_samples = self.encode_recdata_v2(samples, ids_order=id_orders)
        sample_embeds, atts_samples, replaced_idx = self.recprompt_wrap_v2(samples_encode, samples, atts_samples,
                                                                           prompt)

        # Construct samples_encode with node_ids and replaced_idx
        samples_encode_out = {
            'node_ids': samples_encode['node_ids'],
            'replaced_idx': replaced_idx,
            'mf_embeddings': samples_encode['mf_embeddings'] if 'mf_embeddings' in samples_encode else None
        }

        # print("prompt_based_encode_v2: samples_encode keys:", samples_encode_out.keys())
        # print("node_ids shape:", samples_encode_out['node_ids'].shape)
        # print("replaced_idx shape:", samples_encode_out['replaced_idx'].shape)
        return sample_embeds, atts_samples, samples_encode_out

    def forward_v1(self, samples):
        samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)
        self.llama_tokenizer.padding_side = "right"
        device = samples['UserID'].device
        ans_ = {1: self.pos_ans, 0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples["label"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
            -100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)
        with self.maybe_autocast():
            if self.use_lora:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        loss = outputs.loss
        return {"loss": loss}

    @projector_regularized_forward
    def forward_v2(self, samples):
        user_selective_prompts = False
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_with_p([5, 5, 5, 1]))
            sample_embeds, atts_samples, samples_encode = self.prompt_based_encode_v2(prompt, samples)

        self.llama_tokenizer.padding_side = "right"
        device = samples['UserID'].device
        ans_ = {1: self.pos_ans[0], 0: self.neg_ans[0]}
        text = [ans_[int(t)] for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones(
            [atts_samples.shape[0], atts_samples.shape[1]],
            dtype=torch.long
        ).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        if not hasattr(self, 'adj_matrix_coo'):
            raise ValueError("Adjacency matrix not loaded in model")
        if 'UserID' not in samples or 'TargetItemID' not in samples:
            raise ValueError("samples must contain 'UserID' and 'TargetItemID' for GCN")
        if 'node_ids' not in samples_encode or 'replaced_idx' not in samples_encode:
            raise ValueError("samples_encode must contain 'node_ids' and 'replaced_idx' for GCN enhancement")

        with self.maybe_autocast():
            fb = self.compute_feedback_gcn(
                samples,
                samples_encode,
                samples_encode['replaced_idx'],
                inputs_embeds,
                attention_mask
            )

            gcn_output = fb['gcn_output']
            # Keep the original train-path post-processing exactly as in the
            # reproduced GraphLoRA baseline.
            gcn_output = F.normalize(gcn_output, dim=-1) * 0.1

            # relation-aware pair vector (step 2/3): added to the user/target
            # node rows AFTER the normalize*x0.1 so c_ui enters in the final
            # injection space; zero-init => identical to baseline at start
            if self.rel_inject and 'pair_add' in fb:
                pa = fb['pair_add']
                gcn_output = gcn_output.float()
                idx = torch.cat([pa['u_node'], pa['t_node']])
                inc = torch.cat([pa['c'], pa['c']], dim=0).to(gcn_output.dtype)
                gcn_output = gcn_output.index_add_(0, idx, inc).to(dtype=torch.float16)

            gcn_output = gcn_output.to(dtype=torch.float16)

            # This is the requested gradient gate:
            # False cuts L_rec -> GNN while preserving the forward injection.
            gcn_output_for_injection = (
                gcn_output
                if self.train_gnn_with_rec_loss
                else gcn_output.detach()
            )

            batch_indices = fb['batch_indices']
            token_positions = fb['token_positions']
            local_node_ids = fb['local_node_ids']

            if self.use_lora:
                # A1 injection warm-up: training only, linear ramp of the
                # effective gnn_scale from 0 to full over the first
                # ratio*total_steps global steps. Eval keeps factor 1.0 so the
                # evaluation protocol stays identical to I3 at every point.
                warmup_factor = 1.0
                if self.training and self.inj_warmup_ratio > 0:
                    _ws = self.inj_warmup_ratio * self.inj_warmup_total_steps
                    if _ws > 0:
                        warmup_factor = min(1.0, float(fb.get("step", 0)) / _ws)
                pair_gate = fb.get('pair_rankgate', None) \
                    if (self.rel_rank_mod and 'pair_rankgate' in fb) else None
                for my_linear in self.my_linears:
                    my_linear.gcn_output = gcn_output_for_injection
                    my_linear.batch_indices = batch_indices
                    my_linear.token_positions = token_positions
                    my_linear.local_node_ids = local_node_ids
                    my_linear.rel_pair_gate = pair_gate['g_raw'] if pair_gate is not None else None
                    my_linear.warmup_factor = warmup_factor

            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )

            if self.use_lora:
                for my_linear in self.my_linears:
                    my_linear.gcn_output = None
                    my_linear.batch_indices = None
                    my_linear.token_positions = None
                    my_linear.local_node_ids = None

        pos_ans_id = self.llama_tokenizer(ans_[1], add_special_tokens=False).input_ids[0]
        logits = outputs.logits[:, -t_posi, :][:, pos_ans_id]
        loss_rec = F.binary_cross_entropy_with_logits(
            logits,
            samples['label'].float()
        )

        loss = loss_rec
        ret = {"loss": loss, "loss_rec": loss_rec.detach()}

        if self.training and self.gnn_semantic_align:
            if self.sem_adapter_on and not getattr(self, 'sem_structural_skip', False):
                # Anchor-Preserving corr loss (R2 series). R0-S structural
                # control (sem_structural_skip: True) falls through to the
                # OLD L_sem below with the adapter module present-but-
                # unwired: the computation is then EXACTLY the no-adapter
                # baseline, so any bit difference vs the I3 retrain isolates
                # the structural effect of the module's existence.
                # Anchor-Preserving Semantic Refinement: L_corr replaces the
                # old unanchored L_sem
                loss_corr = self._compute_corr_loss(fb)
                loss = loss_rec + self.sem_corr_weight * loss_corr
                ret["loss"] = loss
                ret["loss_sem"] = loss_corr.detach()
            else:
                loss_sem = self._compute_gnn_semantic_loss(fb)
                loss = loss_rec + self.gnn_semantic_weight * loss_sem
                ret["loss"] = loss
                ret["loss_sem"] = loss_sem.detach()

            self._semantic_step += 1
            if self._semantic_step % 100 == 0:
                print(
                    f"[semantic] step={self._semantic_step} "
                    f"L_rec={loss_rec.detach().item():.6f} "
                    f"L_corr={ret['loss_sem'].item():.6f} "
                    f"corr_w={self.sem_corr_weight if self.sem_adapter_on else self.gnn_semantic_weight:.6f} "
                    f"alpha={self.sem_alpha} rho={self.sem_rho} "
                    f"in_path={self.sem_adapter_in_path} trunk_grad={self.sem_trunk_grad} "
                    f"rec->gnn={self.train_gnn_with_rec_loss}",
                    flush=True
                )

        if self.training and self.pref_distill and 'cand_group' in samples:
            loss_pref = self._pref_distill_loss(fb, logits, samples)
            loss = loss + self.pref_distill_weight * loss_pref
            ret["loss"] = loss
            ret["loss_pref"] = loss_pref.detach()
            if self.gnn_conf_loss_weight > 0:
                loss_conf = self._gnn_conf_loss(fb, samples)
                loss = loss + self.gnn_conf_loss_weight * loss_conf
                ret["loss"] = loss
                ret["loss_conf"] = loss_conf.detach()

        return ret


    def generate_for_samples_v1(self, samples):
        samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)
        self.llama_tokenizer.padding_side = "right"
        device = samples_encode['User_emb'].device
        ans_ = {1: self.pos_ans, 0: self.neg_ans}
        text = [random.choice(ans_[int(t)]) + self.end_sym for t in samples["label"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
            -100)
        targets = torch.cat([empty_targets, targets], dim=1)
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)
        with self.maybe_autocast():
            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        loss = outputs.loss
        return {"loss": loss}

    def generate_for_samples_v2(self, samples, return_all=False):
        mode = 'train'
        if 'is_valid' in samples and samples['is_valid']:
            mode = 'valid'
        elif 'is_test' in samples and samples['is_test']:
            mode = 'test'
        self._load_adjacency_matrix(mode=mode)

        user_selective_prompts = False
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_with_p([5, 5, 5, 1]))
            id_orders = get_ids_order(prompt)
            samples_encode, atts_samples = self.encode_recdata_v2(samples, id_orders)
            sample_embeds, atts_samples, replaced_idx = self.recprompt_wrap_v2(
                samples_encode, samples, atts_samples, prompt
            )

        self.llama_tokenizer.padding_side = "right"
        device = samples['UserID'].device
        pos_ans = self.pos_ans[0]
        neg_ans = self.neg_ans[0]
        ans_ = {1: pos_ans, 0: neg_ans}
        text = [ans_[int(t)] for t in samples["label"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)

        t_posi = to_regress_tokens.input_ids.shape[-1] + 1
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones(
            [atts_samples.shape[0], atts_samples.shape[1]],
            dtype=torch.long
        ).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_samples, to_regress_tokens.attention_mask], dim=1)

        if not hasattr(self, 'adj_matrix_coo'):
            raise ValueError("Adjacency matrix not loaded in model")
        if 'UserID' not in samples or 'TargetItemID' not in samples:
            raise ValueError("samples must contain 'UserID' and 'TargetItemID' for GCN")

        with self.maybe_autocast():
            if self.use_lora:
                self.llama_model_lora.eval()
                fb = self.compute_feedback_gcn(
                    samples,
                    samples_encode,
                    replaced_idx,
                    inputs_embeds,
                    attention_mask
                )
                gcn_output = fb['gcn_output'].to(dtype=torch.float16)
                batch_indices = fb['batch_indices']
                token_positions = fb['token_positions']
                local_node_ids = fb['local_node_ids']

                # relation-aware pair add (step 2/3) must apply at eval too,
                # otherwise the eval-time injection distribution differs from
                # the trained one
                if self.rel_inject and 'pair_add' in fb:
                    pa = fb['pair_add']
                    gcn_output = gcn_output.float()
                    idx = torch.cat([pa['u_node'], pa['t_node']])
                    inc = torch.cat([pa['c'], pa['c']], dim=0).to(gcn_output.dtype)
                    gcn_output = gcn_output.index_add_(0, idx, inc).to(dtype=torch.float16)

                for my_linear in self.my_linears:
                    my_linear.gcn_output = gcn_output
                    my_linear.batch_indices = batch_indices
                    my_linear.token_positions = token_positions
                    my_linear.local_node_ids = local_node_ids
                    # eval protocol must stay identical to I3 regardless of
                    # the training-side warm-up schedule
                    my_linear.warmup_factor = 1.0
                    # rank-wise pair gate (step 4): must also be present at
                    # eval — a stale training-step gate indexed by the eval
                    # batch rows caused a CUDA OOB abort in N5
                    my_linear.rel_pair_gate = fb['pair_rankgate']['g_raw'] \
                        if 'pair_rankgate' in fb else None

            if not self.use_lora:
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            else:
                outputs = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )

            if self.use_lora:
                for my_linear in self.my_linears:
                    my_linear.gcn_output = None
                    my_linear.batch_indices = None
                    my_linear.token_positions = None
                    my_linear.local_node_ids = None

            pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
            logits = outputs.logits[:, -t_posi, :][:, pos_ans_id]
            loss = F.binary_cross_entropy_with_logits(
                logits,
                samples['label'].float()
            )

            if return_all:
                return {"outputs": outputs, "logits": logits, "loss": loss}
            return {"loss": loss, "logits": logits}


    def generate_for_samples(self, samples):
        if self.run_mode_ == 'v1':
            return self.generate_for_samples_v1(samples)
        elif self.run_mode_ == 'v2':
            return self.generate_for_samples_v2(samples)
        else:
            raise NotImplementedError("Not implement the default version")

    def generate_sequence(self, samples):
        # 动态加载邻接矩阵基于样本的模式
        mode = 'train'
        if 'is_valid' in samples and samples['is_valid']:
            mode = 'valid'
        elif 'is_test' in samples and samples['is_test']:
            mode = 'test'
        self._load_adjacency_matrix(mode=mode)
        # print(f"[generate_sequence] Loaded adjacency matrix for mode: {mode}")

        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            id_orders = get_ids_order(prompt)
            samples_encode, atts_samples = self.encode_recdata_v2(samples, id_orders)
            sample_embeds, atts_samples, replaced_idx = self.recprompt_wrap_v2(samples_encode, samples, atts_samples,
                                                                               prompt)
        inputs_embeds = sample_embeds

        if not hasattr(self, 'adj_matrix_coo'):
            raise ValueError("Adjacency matrix not loaded in model")

        if 'UserID' not in samples or 'TargetItemID' not in samples:
            raise ValueError("samples must contain 'UserID' and 'TargetItemID' for GCN")
        node_ids = torch.cat([
            samples['UserID'],
            samples['TargetItemID'] + self.num_users
        ], dim=0).to(self.device)

        # 提取子图
        sub_adj_matrix, sub_node_ids = self.get_subgraph(node_ids)
        # print(f"[generate_sequence] sub_adj_matrix shape: {sub_adj_matrix.shape}")
        # print(f"[generate_sequence] sub_node_ids shape: {sub_node_ids.shape}")

        with torch.no_grad():
            try:
                if not self.use_lora:
                    outputs = self.llama_model.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=10,
                        num_beams=1,
                        do_sample=True,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.0,
                        length_penalty=1,
                        temperature=1.0,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                else:
                    batch_size = samples['UserID'].shape[0]
                    node_features = samples_encode['mf_embeddings']
                    with torch.cuda.amp.autocast(enabled=False):
                        sub_adj_matrix = sub_adj_matrix.to(dtype=torch.float32, device=self.device)
                        node_features = node_features[sub_node_ids].to(dtype=torch.float32, device=self.device)
                        gcn_output = self.gcn(sub_adj_matrix, node_features, sub_node_ids)
                        # print(f"[generate_sequence] gcn_output shape: {gcn_output.shape}")
                    original_inputs_embeds = inputs_embeds.clone()
                    outputs = self.llama_model_lora.generate(
                        inputs_embeds=inputs_embeds,
                        max_new_tokens=10,
                        num_beams=1,
                        do_sample=True,
                        min_length=1,
                        top_p=0.9,
                        repetition_penalty=1.0,
                        length_penalty=1,
                        temperature=1.0,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    node_map = {old_id.item(): new_id for new_id, old_id in enumerate(sub_node_ids)}
                    # print(f"[generate_sequence] node_map: {list(node_map.items())[:5]}")
                    for name, module in self.llama_model_lora.named_modules():
                        if hasattr(module, 'lora_A') and any(t in name for t in ["q_proj", "v_proj"]):  #, "k_proj", "o_proj"
                            lora_a_output = module.lora_A[module.active_adapter](
                                module.lora_dropout[module.active_adapter](original_inputs_embeds)
                            )
                            for b in range(batch_size):
                                batch_mask = replaced_idx[:, 0] == b
                                if not batch_mask.any():
                                    continue
                                batch_replaced_idx = replaced_idx[batch_mask, 1]
                                batch_node_ids = samples_encode['node_ids'][batch_mask]
                                for token_pos, node_id in zip(batch_replaced_idx, batch_node_ids):
                                    local_node_id = node_map.get(node_id.item(), -1)
                                    if local_node_id == -1:
                                        print(
                                            f"[generate_sequence] Warning: node_id {node_id} not found in sub_node_ids")
                                        continue
                                    lora_a_output[b, token_pos] += gcn_output[local_node_id]
            except Exception as e:
                print(f"errors.....: {e}")
        print(inputs_embeds.shape, outputs.sequences.shape)
        print(self.llama_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True), samples['label'])
        print()
        return {"loss": 0, 'logits': outputs.logit}

    def encode_allinputs(self, samples, mode='v1'):
        if mode == 'v2':
            samples_encode, atts_samples = self.encode_recdata_v2(samples)
        else:
            samples_encode, atts_samples = self.encode_recdata_v1(samples)
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            if mode == 'v2':
                sample_embeds, atts_samples, _ = self.recprompt_wrap_v2(samples_encode, samples, atts_samples, prompt)
            else:
                sample_embeds, atts_samples = self.recprompt_wrap_v1(samples_encode, samples, atts_samples, prompt)
        inputs_embeds = sample_embeds
        return inputs_embeds

    @classmethod
    def from_config(cls, cfg):
        rec_model = cfg.get('rec_model', "MF")
        rec_config = cfg.get("rec_config")
        freeze_rec = cfg.get("freeze_rec", True)
        rec_precision = cfg.get("rec_precision", 'fp16')
        lora_config = cfg.get("lora_config")
        llama_model = cfg.get("llama_model")
        proj_mid = cfg.get("proj_mid_times")
        freeze_proj = cfg.get("freeze_proj")
        freeze_lora = cfg.get("freeze_lora")
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        lora_gnn = cfg.get("lora_gnn", "")
        dataset = cfg.get("dataset", "")
        lora_scale = cfg.get("lora_scale", 1.0)
        gnn_scale = cfg.get("gnn_scale", 0)
        gcn_layer = cfg.get("gcn_layer", 0)
        collab_mode = cfg.get("collab_mode", "legacy")

        # Only the requested new controls.
        gnn_semantic_align = cfg.get("gnn_semantic_align", False)
        gnn_semantic_weight = cfg.get("gnn_semantic_weight", 0.05)
        train_gnn_with_rec_loss = cfg.get("train_gnn_with_rec_loss", True)

        # Preference-level distillation controls (v12 recipe).
        pref_distill = cfg.get("pref_distill", False)
        pref_distill_weight = cfg.get("pref_distill_weight", 0.5)
        pref_tau_l = cfg.get("pref_tau_l", 1.0)
        pref_tau_g = cfg.get("pref_tau_g", 1.0)
        pref_kappa_init = cfg.get("pref_kappa_init", 5.0)
        gnn_conf_loss_weight = cfg.get("gnn_conf_loss_weight", 0.0)
        rel_inject = cfg.get("rel_inject", False)
        rel_gate = cfg.get("rel_gate", False)
        rel_rank_mod = cfg.get("rel_rank_mod", False)
        sem_adapter = cfg.get("sem_adapter", False)
        sem_alpha = cfg.get("sem_alpha", 0.0)
        sem_rho = cfg.get("sem_rho", 0.05)
        sem_corr_weight = cfg.get("sem_corr_weight", 0.04)
        sem_adapter_in_path = cfg.get("sem_adapter_in_path", True)
        sem_trunk_grad = cfg.get("sem_trunk_grad", False)
        sem_structural_skip = cfg.get("sem_structural_skip", False)
        sem_decay = cfg.get("sem_decay", False)
        sem_decay_steps = cfg.get("sem_decay_steps", "11297,22594")
        grad_clip = cfg.get("grad_clip", 0.0)
        inj_warmup_start = cfg.get("inj_warmup_start", 0)
        inj_warmup_len = cfg.get("inj_warmup_len", 0)
        # A0/A1 injection controls (see MyLinear / forward_v2)
        inj_diag = cfg.get("inj_diag", False)
        gnn_injection_warmup_ratio = cfg.get("gnn_injection_warmup_ratio", 0.0)
        gnn_injection_warmup_total_steps = cfg.get("gnn_injection_warmup_total_steps", 0)
        # CW series: confidence-weighted consensus semantic teacher
        sem_consensus = cfg.get("sem_consensus", False)
        sem_consensus_layers = cfg.get("sem_consensus_layers", "6,8,10")
        sem_conf_tau = cfg.get("sem_conf_tau", 0.001)
        sem_consensus_target = cfg.get("sem_consensus_target", "mean")
        lora_rank_groups = cfg.get("lora_rank_groups", None)
        fix_autocast_cache = cfg.get("fix_autocast_cache", False)

        model = cls(
            rec_model=rec_model,
            rec_config=rec_config,
            pretrained_rec=rec_config['pretrained_path'],
            freeze_rec=freeze_rec,
            rec_precision=rec_precision,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            proj_token_num=cfg.get("proj_token_num"),
            proj_drop=cfg.get("proj_drop"),
            lora_config=lora_config,
            proj_mid=proj_mid,
            proj_reg_weight=cfg.get("proj_reg_weight", 0.0),
            proj_reg_items=cfg.get("proj_reg_items", 64),
            freeze_lora=freeze_lora,
            freeze_proj=freeze_proj,
            lora_gnn=lora_gnn,
            lora_scale=lora_scale,
            gnn_scale=gnn_scale,
            gcn_layer=gcn_layer,
            collab_mode=collab_mode,
            gnn_semantic_align=gnn_semantic_align,
            gnn_semantic_weight=gnn_semantic_weight,
            train_gnn_with_rec_loss=train_gnn_with_rec_loss,
            pref_distill=pref_distill,
            pref_distill_weight=pref_distill_weight,
            pref_tau_l=pref_tau_l,
            pref_tau_g=pref_tau_g,
            pref_kappa_init=pref_kappa_init,
            gnn_conf_loss_weight=gnn_conf_loss_weight,
            rel_inject=rel_inject,
            rel_gate=rel_gate,
            rel_rank_mod=rel_rank_mod,
            sem_adapter=sem_adapter,
            sem_alpha=sem_alpha,
            sem_rho=sem_rho,
            sem_corr_weight=sem_corr_weight,
            sem_adapter_in_path=sem_adapter_in_path,
            sem_trunk_grad=sem_trunk_grad,
            sem_structural_skip=sem_structural_skip,
            sem_decay=sem_decay,
            sem_decay_steps=sem_decay_steps,
            grad_clip=grad_clip,
            inj_warmup_start=inj_warmup_start,
            inj_warmup_len=inj_warmup_len,
            inj_diag=inj_diag,
            gnn_injection_warmup_ratio=gnn_injection_warmup_ratio,
            gnn_injection_warmup_total_steps=gnn_injection_warmup_total_steps,
            sem_consensus=sem_consensus,
            sem_consensus_layers=sem_consensus_layers,
            sem_conf_tau=sem_conf_tau,
            sem_consensus_target=sem_consensus_target,
            lora_rank_groups=lora_rank_groups,
            fix_autocast_cache=fix_autocast_cache,
            dataset=dataset,
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load MiniGPT4Rec Checkpoint: {}".format(ckpt_path))
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print("loading message, msg.... ", msg)
            del ckpt
            if os.path.exists(rec_config['pretrained_path']) and freeze_rec:
                model.rec_encoder.load_state_dict(
                    torch.load(rec_config['pretrained_path'], map_location="cpu")
                )

        ans_type = cfg.get('ans_type')
        model.set_answer_type(mode=ans_type)
        model.print_prompt()
        return model

