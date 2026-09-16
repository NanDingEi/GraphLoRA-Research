import logging
import random
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

    def forward(self, adj_matrix, node_features, node_ids=None, return_penult=False):
        h = node_features
        penult = None
        for i, layer in enumerate(self.layers):
            if return_penult and i == len(self.layers) - 1:
                # 256-d embedding feeding the last (8-d) layer, before the
                # final normalize*0.1 / fp16 cast used for LoRA injection
                penult = F.normalize(h, dim=-1)
            h = layer(adj_matrix, h)
        output = F.normalize(h, dim=-1) * 0.1
        output = output.to(dtype=torch.float16)
        if return_penult:
            return output, penult
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
    def __init__(self, original_linear, layer_idx=None, lora_scale=1.0, gnn_scale=0 ,gcn_layer=0):
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
        # GCN data attributes
        self.cf_m = None   # (B, r, r) graph-conditioned rank interaction (cflora)
        self.gcn_output = None
        self.batch_indices = None
        self.token_positions = None
        self.local_node_ids = None

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        # Step 1: Compute the base linear transformation
        weight = transpose(self.weight, self.fan_in_fan_out)
        result = F.linear(x, weight, bias=self.bias)

        # Step 2: Check if LoRA is applicable
        if self.active_adapter not in self.lora_A.keys():
            return result.to(previous_dtype)

        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            return result.to(previous_dtype)

        elif self.r[self.active_adapter] > 0 and not self.merged:

            x_lora = x.to(self.lora_A[self.active_adapter].weight.dtype)
            # print(f"[MyLinear] Layer {self.layer_idx} - x_lora shape: {x_lora.shape}")

            x_lora = self.lora_dropout[self.active_adapter](x_lora)
            # print(f"[MyLinear] Layer {self.layer_idx} - x_lora after dropout shape: {x_lora.shape}")

            lora_a_output = self.lora_A[self.active_adapter](x_lora)
            # print(f"[MyLinear] Layer {self.layer_idx} - lora_a_output shape: {lora_a_output.shape}")

            if self.layer_idx == self.gcn_layer and self.cf_m is not None:
                # Graph-conditioned Fully-Interactive LoRA (ReLLaX-style):
                # r' = (I + M_G) r at ID-token positions, then B r'
                # M_G (per-sample, zero-init projector) re-weights the interaction
                # between LoRA rank components from the collaborative state
                r_vec = lora_a_output[self.batch_indices, self.token_positions]      # (N, r)
                mod = r_vec + torch.einsum('bij,bj->bi',
                                           self.cf_m[self.batch_indices].to(r_vec.dtype), r_vec)
                lora_a_output[self.batch_indices, self.token_positions] = mod
            elif self.layer_idx == self.gcn_layer and self.gcn_output is not None:

                valid_mask = self.local_node_ids != -1
                # print(
                #     f"[MyLinear] Layer {self.layer_idx} - valid_mask shape: {valid_mask.shape}, count: {valid_mask.sum()}")

                if valid_mask.any():
                    # 3.4.3: Extract valid indices and positions
                    batch_indices = self.batch_indices[valid_mask]
                    token_positions = self.token_positions[valid_mask]
                    local_node_ids = self.local_node_ids[valid_mask]
                    # print(f"[MyLinear] Layer {self.layer_idx} - batch_indices shape: {batch_indices.shape}")
                    # print(f"[MyLinear] Layer {self.layer_idx} - token_positions shape: {token_positions.shape}")
                    # print(f"[MyLinear] Layer {self.layer_idx} - local_node_ids shape: {local_node_ids.shape}")
                    # print(f"[MyLinear] Layer {self.layer_idx} - gcn_output shape: {self.gcn_output.shape}")

                    # lora_a_mean_before = lora_a_output[batch_indices, token_positions].abs().mean()
                    # print(
                    #     f"[MyLinear] Layer {self.layer_idx} - lora_a_output mean before GCN: {lora_a_mean_before}")

                    lora_a_output[batch_indices, token_positions] = (
                            lora_a_output[batch_indices, token_positions] * self.lora_scale +
                            self.gcn_output[local_node_ids] * self.gnn_scale
                    )

                    # lora_a_mean_after = lora_a_output[batch_indices, token_positions].abs().mean()
                    # print(f"[MyLinear] Layer {self.layer_idx} - lora_a_output mean after GCN: {lora_a_mean_after}")
                    # print(
                    #     f"[MyLinear] Layer {self.layer_idx} - lora_a_output shape after GCN: {lora_a_output.shape}")

            lora_output = self.lora_B[self.active_adapter](lora_a_output)
            # print(f"[MyLinear] Layer {self.layer_idx} - lora_b_output shape: {lora_b_output.shape}")

            result += lora_output
            # print(f"[MyLinear] Layer {self.layer_idx} - result shape after adding lora_output: {result.shape}")

        result = result.to(previous_dtype)
        # print(f"[MyLinear] Layer {self.layer_idx} - final result shape: {result.shape}")

        return result


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
            gcn_layer =0,
            semantic_feedback=False,
            semantic_layer=8,
            feedback_detach=False,
            feedback_gate_bias_init=-2.0,
            mutual_mode="off",
            mutual_layer=8,
            lambda_llm_to_gnn=0.005,
            lambda_gnn_to_llm=0.02,
            rel_temp_llm=1.0,
            rel_temp_gnn=1.0,
            b2_temp=0.1,
            mutual_rank_loss=False,
            rank_margin=0.02,
            rank_weight_ref=0.05,
            mutual_warmup_iters=0,
            mutual_gate="off",
            gate_tau_high=0.75,
            gate_tau_low=0.6,
            gate_delta=0.1,
            gnn_conf_kappa=5.0,
            gnn_conf_loss_weight=0.1,
            injection_mode="legacy",
            injection_layers=None,
            router_entropy_weight=0.0,
            expert_init_scale=0.02,
            pref_distill_weight=0.0,
            pref_tau_l=1.0,
            pref_tau_g=1.0,
            eval_fuse_beta=0.0,
            pref_tca=False,
            tca_alpha_max=0.2,
            tca_beta_max=0.1,
            tca_warmup_iters=500,
            rel_ema_momentum=0.0,
            grad_projection=False,
            rel_topk=0,
            sem_refine=False,
            rlm_bridge_dim=256,
            rlm_tau=0.1
    ):
        super().__init__()
        self.low_resource = low_resource
        self.proj_token_num = proj_token_num
        self.dataset = dataset
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
        if lora_config is not None and lora_config.use_lora:
            print("Setting LoRA")
            self.use_lora = True
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],  #,"k_proj", "o_proj"
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model_lora = get_peft_model(self.llama_model, peft_config)

            if lora_gnn == "lightgcn":
                self.gcn = LightGCN(in_dim=256, out_dim=lora_config.r, num_layers=3, dropout=0.1 )
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
                    f"Unsupported lora_gnn type: {lora_gnn}. Choose from ['lightgcn', 'ngcf', 'graphsage', 'gat' , 'gcn']")
            self.num_users = rec_config.user_num
            self.num_items = rec_config.item_num
            # ---- B9/B10: injection channel selection ----
            assert injection_mode in ("legacy", "expert", "cflora"), \
                f"unsupported injection_mode: {injection_mode}"
            self.injection_mode = injection_mode
            self.router_entropy_weight = float(router_entropy_weight)
            if injection_mode == "expert":
                # B9/B10: the collaborative signal enters the hidden space
                # through its own expert (P_G: 256->r, B_G: r->hidden, zero-init
                # so a warm-started model is bit-identical at step 0), scaled by
                # a per-token router rho and a per-layer router alpha. Injection
                # happens via forward hooks on decoder layers, so no MyLinear
                # replacement is needed.
                if isinstance(injection_layers, str):
                    layers = [int(x) for x in injection_layers.split(",")]
                elif injection_layers is None:
                    layers = [int(gcn_layer)]
                else:
                    layers = [int(x) for x in injection_layers]
                num_decoder_layers = len(self.llama_model_lora.base_model.model.model.layers)
                assert all(0 <= l < num_decoder_layers for l in layers), \
                    f"injection_layers {layers} out of range ({num_decoder_layers} layers)"
                self.injection_layers = layers
                hidden = self.llama_model.config.hidden_size
                d = self.rec_encoder.config.embedding_size
                r = lora_config.r
                self.collab_A = nn.ModuleList()   # P_G_l: 256 -> r
                self.collab_B = nn.ModuleList()   # B_G_l: r -> hidden, zero-init
                self.inj_router = nn.ModuleList() # rho_l: [h; z] -> scalar
                for _ in layers:
                    a = nn.Linear(d, r)
                    b = nn.Linear(r, hidden)
                    # random direction init: magnitude is set by the relative-norm
                    # scaling in the hook, but the direction must be nonzero
                    # (a zero-init B_G would give dh = 0/||dh|| = 0 forever)
                    nn.init.normal_(b.weight, std=0.02)
                    nn.init.zeros_(b.bias)
                    rt = nn.Linear(hidden + d, 1)
                    # small std so the logit starts near 0 (rho ~ 0.5): the
                    # 4096-d hidden state has a large norm and default init
                    # would saturate the sigmoid to 0, killing the injection
                    nn.init.normal_(rt.weight, std=0.01)
                    nn.init.zeros_(rt.bias)
                    self.collab_A.append(a)
                    self.collab_B.append(b)
                    self.inj_router.append(rt)
                # B10 layer router alpha: softmax over candidate layers, driven
                # by the user's collaborative representation z_u
                self.layer_router = nn.Linear(d, len(layers))
                # relative injection scale: add = gamma * ||h|| * dh/||dh|| so the
                # expert's contribution is a fixed FRACTION of the token hidden
                # norm — visible in fp16 (absolute adds below the fp16 ulp of the
                # ~100-magnitude hidden states get rounded away, which would
                # give the zero-init expert dead gradients forever)
                self.expert_scale = nn.Parameter(torch.tensor(float(expert_init_scale)))
                self._injection_on = False
                self._inj = None
                decoder_layers = self.llama_model_lora.base_model.model.model.layers
                self._inj_handles = [
                    decoder_layers[l].register_forward_hook(self._make_injection_hook(slot))
                    for slot, l in enumerate(layers)
                ]
                print(f"B9/B10 expert injection ON at layers {layers} "
                      f"(relative-norm experts scale={expert_init_scale}, learned rho/alpha routers)")
            else:
                self.injection_layers = []
                self._injection_on = False
                self._inj = None

            if injection_mode == "cflora":
                # ReLLaX-style Graph-conditioned Fully-Interactive LoRA:
                # collaborative state h_G -> projector -> M_G (r x r) ->
                # DeltaW = B (I + M_G) A  (zero-init projector => warm-start safe)
                d = self.rec_encoder.config.embedding_size
                r_cf = int(lora_config.r)
                self.cf_r = r_cf
                self.interaction_projector = nn.Sequential(
                    nn.Linear(3 * d, 4 * d),
                    nn.GELU(),
                    nn.Linear(4 * d, r_cf * r_cf),
                )
                nn.init.zeros_(self.interaction_projector[-1].weight)
                nn.init.zeros_(self.interaction_projector[-1].bias)
                self.cf_m = None
                print(f"Graph-conditioned CFLoRA ON: projector {3*d}->4d->{r_cf*r_cf}, "
                      f"zero-init (warm-start safe), replaces 8-d hidden injection")
            if injection_mode in ("legacy", "cflora"):
                # Store MyLinear instances
                target_layer_str = f'model.layers.{gcn_layer}'
                print(f"Preparing to inject GCN into: {target_layer_str} with gnn_scale: {gnn_scale}")

                # Store MyLinear instances
                self.my_linears = []
                for name, module in self.llama_model_lora.named_modules():
                    # 2. 使用 target_layer_str 替代硬编码的 'model.layers.31'
                    if target_layer_str in name and any(t in name for t in ['q_proj', 'v_proj']):   #, 'k_proj', 'o_proj'
                        if isinstance(module, OriginalLinear):
                            # 3. 实例化 MyLinear 时传入所有动态参数
                            # 注意：这里必须传入 gnn_scale 和 gcn_layer，否则 MyLinear 里这些值默认为 0
                            new_module = MyLinear(
                                original_linear=module,
                                layer_idx=gcn_layer,
                                lora_scale=lora_scale,
                                gnn_scale=gnn_scale,
                                gcn_layer=gcn_layer
                            )

                            # 替换模型中的层
                            parent_name = '.'.join(name.split('.')[:-1])
                            attr_name = name.split('.')[-1]
                            parent_module = self.llama_model_lora
                            for part in parent_name.split('.'):
                                parent_module = getattr(parent_module, part)
                            setattr(parent_module, attr_name, new_module)

                            self.my_linears.append(new_module)

                print(
                    f"Setting LoRA, GCN, and MyLinear for layer {gcn_layer} Done. Replaced {len(self.my_linears)} modules.")

            # ---- BiLoop closed-loop semantic feedback (LLM -> recommender) ----
            # E -> LLM_{0:l_s} -> delta_h -> W_back/gate -> e_tilde -> GNN -> LLM_{l_g:L}
            self.semantic_feedback = bool(semantic_feedback)
            self.semantic_layer = int(semantic_layer)
            self.feedback_detach = bool(feedback_detach)
            if self.semantic_feedback:
                assert self.semantic_layer < gcn_layer, \
                    f"semantic_layer ({self.semantic_layer}) must be < gcn_layer ({gcn_layer}) " \
                    "so the feedback pass never reaches the injection layer"
                hidden = self.llama_model.config.hidden_size
                d = self.rec_encoder.config.embedding_size
                # W_back: LLM hidden -> rec space, zero-init so the loop starts closed (s = 0)
                self.semantic_back_proj = nn.Linear(hidden, d)
                nn.init.zeros_(self.semantic_back_proj.weight)
                nn.init.zeros_(self.semantic_back_proj.bias)
                # W_g: gate on [e; s]; bias init negative so the gate starts mostly closed
                self.semantic_gate = nn.Linear(2 * d, d)
                self.semantic_gate.bias.data.fill_(float(feedback_gate_bias_init))
                print(
                    f"BiLoop semantic feedback ON: semantic_layer={self.semantic_layer}, "
                    f"injection layer={gcn_layer}, back_proj {hidden}->{d}, gate {2*d}->{d}")

            # ---- B2/B3 bidirectional mutual learning (LLM <-> GNN) ----
            # LLM semantic view: hs[mutual_layer] at ID-token positions (pass-1, exit_layer).
            # GNN collaborative view: penultimate 256-d GNN embedding.
            # Neither side is frozen: stop-grad only fixes the *target* inside each
            # auxiliary loss, LoRA keeps updating via L_rec and L_{G->L}.
            assert mutual_mode in ("off", "b2", "b3", "rel", "rlm"), f"unsupported mutual_mode: {mutual_mode}"
            assert not self.semantic_feedback, \
                "semantic_feedback and mutual_mode are mutually exclusive"
            self.mutual_mode = mutual_mode
            self.mutual_layer = int(mutual_layer)
            self.lambda_llm_to_gnn = float(lambda_llm_to_gnn)
            self.lambda_gnn_to_llm = float(lambda_gnn_to_llm)
            self.rel_temp_llm = float(rel_temp_llm)
            self.rel_temp_gnn = float(rel_temp_gnn)
            self.b2_temp = float(b2_temp)
            self.mutual_rank_loss = bool(mutual_rank_loss)
            self.rank_margin = float(rank_margin)
            self.rank_weight_ref = float(rank_weight_ref)
            self.mutual_warmup_iters = int(mutual_warmup_iters)
            # ---- B11: disagreement-driven active mutual learning ----
            assert mutual_gate in ("off", "threshold", "disagreement"), \
                f"unsupported mutual_gate: {mutual_gate}"
            if mutual_gate != "off":
                assert self.mutual_mode == "b3" and self.mutual_rank_loss, \
                    "mutual gating currently requires mutual_mode=b3 + mutual_rank_loss"
            if mutual_gate != "off" or float(gnn_conf_loss_weight) > 0:
                # kappa exists whenever gating OR the confidence-calibration
                # auxiliary is wanted (v6 attribution showed the gains may come
                # from the kappa head itself, so it must be testable alone)
                self.gnn_conf_kappa = nn.Parameter(torch.tensor(float(gnn_conf_kappa)))
            self.mutual_gate = mutual_gate
            self.gate_tau_high = float(gate_tau_high)
            self.gate_tau_low = float(gate_tau_low)
            self.gate_delta = float(gate_delta)
            self.gnn_conf_loss_weight = float(gnn_conf_loss_weight)
            # ---- preference-level G->L distillation (candidates per sample) ----
            self.pref_distill_weight = float(pref_distill_weight)
            self.pref_tau_l = float(pref_tau_l)
            self.pref_tau_g = float(pref_tau_g)
            self.eval_fuse_beta = float(eval_fuse_beta)
            self.sem_refine = False
            # ---- TCA-style prediction-level mutual optimization ----
            self.pref_tca = bool(pref_tca)
            self.tca_alpha_max = float(tca_alpha_max)
            self.tca_beta_max = float(tca_beta_max)
            self.tca_warmup_iters = int(tca_warmup_iters)
            # ---- stabilizers for the mutual cooperation ----
            self.rel_ema_momentum = float(rel_ema_momentum)
            if self.mutual_mode == "rel" and self.rel_ema_momentum > 0:
                n_total = int(rec_config.item_num) + int(rec_config.user_num)
                self.register_buffer("_sem_bank",
                                     torch.zeros(max(n_total, 1),
                                                 self.llama_model.config.hidden_size),
                                     persistent=False)
                print(f"REL EMA teacher ON: momentum={self.rel_ema_momentum}, "
                      f"bank rows={n_total}")
            self.grad_projection = bool(grad_projection)
            self.rel_topk = int(rel_topk)
            # ---- RLMRec-Con: bridge-space cross-view contrastive alignment ----
            self.rlm_bridge_dim = int(rlm_bridge_dim)
            self.rlm_tau = float(rlm_tau)
            if self.mutual_mode == "rlm":
                d_llm = self.llama_model.config.hidden_size
                d_g = self.rec_encoder.config.embedding_size
                # Semantic Bridge Heads: recommendation-oriented views in a
                # shared bridge space (RLMRec's cross-view alignment, adapted
                # to online LLM hidden states instead of offline profiles)
                self.rlm_head_L = nn.Linear(d_llm, self.rlm_bridge_dim)
                self.rlm_head_G = nn.Linear(d_g, self.rlm_bridge_dim)
                print(f"RLMRec-Con ON: bridge heads {d_llm}->{self.rlm_bridge_dim}<-{d_g}, "
                      f"tau={self.rlm_tau}, bidirectional InfoNCE with stop-grad")
            # ---- Semantic Graph Refinement (CLLM4Rec-inspired, scheme 2) ----
            # the LLM proposes candidate semantic edges between batch items
            # (cos of mid-layer hidden states); a learned gate decides whether
            # the graph accepts them; the refined adjacency then feeds the GNN,
            # whose output flows back through GraphLoRA into the LLM
            self.sem_refine = bool(sem_refine) and self.mutual_mode != "off"
            if bool(sem_refine) and self.mutual_mode == "off":
                print("[sem-refine] requires an active mutual mode (pass-1); disabled")
            if self.sem_refine:
                hidden = self.llama_model.config.hidden_size
                d = self.rec_encoder.config.embedding_size
                # symmetric pair features: (h_i+h_j, z_i+z_j) -> acceptance prob
                self.sem_gate = nn.Linear(hidden + d, 1)
                nn.init.normal_(self.sem_gate.weight, std=0.01)
                nn.init.constant_(self.sem_gate.bias, -2.0)   # edges mostly closed at start
                print(f"Semantic Graph Refinement ON: gated semantic edges between "
                      f"batch items (gate bias=-2, learnable acceptance)")
            if self.mutual_mode != "off":
                assert self.mutual_layer < gcn_layer, \
                    f"mutual_layer ({self.mutual_layer}) must be < gcn_layer ({gcn_layer}) " \
                    "so the mutual pass never reaches the injection layer"
                assert lora_gnn in ("ngcf", "lightgcn"), \
                    f"mutual learning requires ngcf/lightgcn backbone, got: {lora_gnn}"
                if self.mutual_mode == "b2":
                    hidden = self.llama_model.config.hidden_size
                    d = self.rec_encoder.config.embedding_size
                    # P_L: 4096 -> 256, P_G: 256 -> 256; default init (new params,
                    # warm-started runs will show them as missing keys — expected)
                    self.mutual_proj_L = nn.Linear(hidden, d)
                    self.mutual_proj_G = nn.Linear(d, d)
                self._mutual_stats = {"n": 0, "l2g": 0.0, "g2l": 0.0}
                print(
                    f"Mutual learning ON: mode={self.mutual_mode}, mutual_layer={self.mutual_layer}, "
                    f"lambda_l2g={self.lambda_llm_to_gnn}, lambda_g2l={self.lambda_gnn_to_llm}")
        else:
            self.gcn = None
            self.semantic_feedback = False
            self.semantic_layer = int(semantic_layer)
            self.feedback_detach = False
            self.mutual_mode = "off"
            self.mutual_layer = int(mutual_layer)
            self.lambda_llm_to_gnn = 0.0
            self.lambda_gnn_to_llm = 0.0
            self.mutual_rank_loss = False
            self.rank_margin = float(rank_margin)
            self.rank_weight_ref = float(rank_weight_ref)
            self.mutual_warmup_iters = int(mutual_warmup_iters)
            self.mutual_gate = "off"
            self.gate_tau_high = float(gate_tau_high)
            self.gate_tau_low = float(gate_tau_low)
            self.gate_delta = float(gate_delta)
            self.gnn_conf_loss_weight = float(gnn_conf_loss_weight)
            self.injection_mode = "legacy"
            self.injection_layers = []
            self.router_entropy_weight = 0.0
            self.pref_distill_weight = float(pref_distill_weight)
            self.pref_tau_l = float(pref_tau_l)
            self.pref_tau_g = float(pref_tau_g)
            self.eval_fuse_beta = float(eval_fuse_beta)
            self.sem_refine = False
            self.rlm_bridge_dim = int(rlm_bridge_dim)
            self.rlm_tau = float(rlm_tau)
        if freeze_lora and self.use_lora:
            print("freeze LoRA...")
            for name, param in self.llama_model_lora.named_parameters():
                param.requires_grad = False
        if self.rec_encoder is not None and 'prompt' not in rec_model:
            self.llama_proj = nn.Sequential(
                nn.Linear(self.rec_encoder.config.embedding_size,
                          self.rec_encoder.config.embedding_size * int(proj_mid)),
                nn.ReLU(),
                nn.Linear(self.rec_encoder.config.embedding_size * int(proj_mid),
                          self.llama_model.config.hidden_size * self.proj_token_num),
            )
        elif self.rec_encoder is not None and rec_model == "personlized_prompt":
            self.llama_proj = nn.Linear(rec_config.item_num + rec_config.user_num,
                                        self.llama_model.config.hidden_size * self.proj_token_num, bias=False)
        elif self.rec_encoder is not None and rec_model == "soft_prompt":
            self.llama_proj = nn.Linear(2, self.llama_model.config.hidden_size * self.proj_token_num, bias=False)
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
                    'train': "/home/bbx/user_lx/GraphLoRA/dataset/ml-1m/movie_graph_train.npz",
                    'valid': "/home/bbx/user_lx/GraphLoRA/dataset/ml-1m/movie_graph_valid.npz",
                    'test': "/home/bbx/user_lx/GraphLoRA/dataset/ml-1m/movie_graph_test.npz"
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
        """Shared subgraph/GNN stage of forward_v2 and generate_for_samples_v2.

        When self.semantic_feedback is True this also implements the BiLoop
        closed loop: run the first `semantic_layer` LLM layers (pass 1), extract
        the contextual residual of the collaborative tokens,
            delta_h = h^(l_s) - h^(0),
        back-project it to the rec space and gate-update the MF embeddings,
            s = W_back(delta_h);  g = sigmoid(W_g[e; s]);  e_tilde = e + g * s,
        scatter e_tilde into a *copy* of the MF tables, and run the GNN on the
        updated features. The raw gcn_output is returned; callers keep their own
        post-processing (train path normalizes, eval path does not).

        When B2/B3 mutual learning is active (training only) this instead runs a
        mutual pass-1 (exit_layer=mutual_layer) and returns the LLM semantic
        view `h_mutual` (hidden states at ID-token positions) alongside the GNN
        collaborative view `token_gnn` (penultimate 256-d GNN embeddings); the
        callers feed both into compute_mutual_losses. No embeddings are rewritten.
        """
        node_ids = samples_encode['node_ids'].to(self.device)
        sub_adj_matrix, sub_node_ids = self.get_subgraph(node_ids)
        node_map = {old_id.item(): new_id for new_id, old_id in enumerate(sub_node_ids)}

        batch_mask = replaced_idx[:, 0] < samples['UserID'].shape[0]
        batch_indices = replaced_idx[batch_mask, 0]
        token_positions = replaced_idx[batch_mask, 1]
        batch_node_ids = node_ids[batch_mask]
        local_node_ids = torch.tensor([node_map.get(nid.item(), -1) for nid in batch_node_ids],
                                      device=self.device)

        mf_embeddings = torch.cat([self.rec_encoder.user_embedding.weight,
                                   self.rec_encoder.item_embedding.weight], dim=0)

        if self.semantic_feedback:
            # ---- pass 1: early-layer LLM pass on the same inputs_embeds ----
            with self.maybe_autocast():
                out1 = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                    exit_layer=self.semantic_layer,
                    output_hidden_states=True,
                )
            hs = out1.hidden_states  # hs[k] = input to layer k; hs[0] = inputs_embeds
            h_l = hs[self.semantic_layer][batch_indices, token_positions]
            h_0 = hs[0][batch_indices, token_positions]
            dh = (h_l - h_0).float()  # (N, hidden) contextual residual
            if self.feedback_detach:
                dh = dh.detach()

            # ---- back-projection + gated update, always in fp32 ----
            with torch.cuda.amp.autocast(enabled=False):
                s = self.semantic_back_proj(dh)                     # (N, d)
                e_rows = mf_embeddings[batch_node_ids]              # (N, d), grad -> tables
                g = torch.sigmoid(self.semantic_gate(torch.cat([e_rows, s], dim=-1)))
                e_tilde = e_rows + g * s
            # scatter into a copy; never mutate the nn.Embedding tables in place
            mf_updated = mf_embeddings.clone()
            mf_updated = mf_updated.index_copy(0, batch_node_ids,
                                               e_tilde.to(mf_embeddings.dtype))

            if not self.training:
                # debug observability: is the feedback loop opening? (eval only)
                st = getattr(self, "_fb_stats", None)
                if st is None:
                    st = self._fb_stats = {"n": 0, "g": 0.0, "s": 0.0, "e": 0.0}
                st["n"] += 1
                st["g"] += g.mean().item()
                st["s"] += s.norm(dim=-1).mean().item()
                st["e"] += e_rows.norm(dim=-1).mean().item()
                if st["n"] % 500 == 0:
                    print(f"[biloop] eval_calls={st['n']} "
                          f"mean_gate={st['g'] / st['n']:.4f} "
                          f"mean||s||={st['s'] / st['n']:.4f} "
                          f"mean||e||={st['e'] / st['n']:.4f}", flush=True)
        else:
            e_tilde = None
            mf_updated = mf_embeddings

        # B2/B3 mutual learning: pass-1 early-layer LLM extraction for the
        # semantic view. No back-projection / gate / embedding rewriting here —
        # the two views are only coupled through the mutual losses.
        h_mutual = None
        if self._mutual_active() and self.training:
            with self.maybe_autocast():
                out1 = self.llama_model_lora(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    use_cache=False,
                    exit_layer=self.mutual_layer,
                    output_hidden_states=True,
                )
            hs = out1.hidden_states  # hs[k] = input to layer k; hs[0] = inputs_embeds
            h_mutual = hs[self.mutual_layer][batch_indices, token_positions].float()  # (N, hidden)
        elif getattr(self, "sem_refine", False) and not self.training:
            # eval-time pass-1 (no grad) so the semantic-edge gate sees the same
            # context as during training — keeps the refined graph consistent
            with torch.no_grad():
                with self.maybe_autocast():
                    out1 = self.llama_model_lora(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        return_dict=True,
                        use_cache=False,
                        exit_layer=self.mutual_layer,
                        output_hidden_states=True,
                    )
                hs = out1.hidden_states
                h_mutual = hs[self.mutual_layer][batch_indices, token_positions].float()

        need_penult = h_mutual is not None or getattr(self, "injection_mode", "legacy") == "expert" \
            or (getattr(self, "pref_distill_weight", 0.0) > 0 and "cand_group" in samples) \
            or (not self.training and getattr(self, "eval_fuse_beta", 0.0) > 0) \
            or getattr(self, "injection_mode", "legacy") == "cflora" \
            or getattr(self, "sem_refine", False)
        self.gcn.eval()
        with torch.cuda.amp.autocast(enabled=False):
            sub_adj_matrix = sub_adj_matrix.to(dtype=torch.float32, device=self.device)
            node_features = mf_updated[sub_node_ids].to(dtype=torch.float32, device=self.device)
            # ---- Semantic Graph Refinement: pass A on the plain graph gives
            # the gate its collaborative context (e^G before refinement) ----
            sem_context_penult = None
            if getattr(self, "sem_refine", False) and h_mutual is not None:
                _, sem_context_penult = self.gcn(sub_adj_matrix, node_features, sub_node_ids,
                                                 return_penult=True)
                sub_adj_matrix = self._refine_adjacency(
                    sub_adj_matrix, sub_node_ids, node_map,
                    node_ids, h_mutual, sem_context_penult)
            if need_penult:
                gcn_output, gcn_penult = self.gcn(sub_adj_matrix, node_features, sub_node_ids,
                                                  return_penult=True)
            else:
                gcn_output = self.gcn(sub_adj_matrix, node_features, sub_node_ids)

        # ---- Graph-conditioned CFLoRA: per-row h_G -> M_G (r x r) ----
        cf_m = None
        if getattr(self, "injection_mode", "legacy") == "cflora":
            d = self.rec_encoder.config.embedding_size
            B_rows = int(batch_indices.max().item()) + 1 if batch_indices.numel() else 0
            h_G = node_features.new_zeros(B_rows, 3 * d)
            for b in range(B_rows):
                tok_rows = (batch_indices == b).nonzero(as_tuple=True)[0]
                if tok_rows.numel() == 0:
                    continue
                nodes_b = node_ids[tok_rows]
                u_gid = int(nodes_b[0].item())
                t_gid = int(nodes_b[-1].item())
                h_gids = [int(x.item()) for x in nodes_b[1:-1]]
                u_loc = node_map.get(u_gid, -1)
                t_loc = node_map.get(t_gid, -1)
                z_u = gcn_penult[u_loc] if u_loc >= 0 else node_features.new_zeros(d)
                z_t = gcn_penult[t_loc] if t_loc >= 0 else node_features.new_zeros(d)
                h_locs = [node_map.get(x, -1) for x in h_gids]
                h_locs = [x for x in h_locs if x >= 0]
                z_h = gcn_penult[h_locs].mean(0) if h_locs else node_features.new_zeros(d)
                h_G[b] = torch.cat([z_u, z_h, z_t])
            cf_m = self.interaction_projector(h_G).reshape(B_rows, self.cf_r, self.cf_r)
            for _ml in getattr(self, "my_linears", []):
                _ml.cf_m = cf_m
                _ml.batch_indices = batch_indices      # token-level (N,)
                _ml.token_positions = token_positions  # token-level (N,)

        ret = {"gcn_output": gcn_output,
               "batch_indices": batch_indices,
               "token_positions": token_positions,
               "local_node_ids": local_node_ids,
               "e_tilde": e_tilde,
               "cf_m": cf_m}
        if need_penult:
            # per-token rows of the collaborative view, aligned with batch_indices
            valid_ln = local_node_ids >= 0
            token_gnn = torch.zeros((local_node_ids.shape[0], gcn_penult.shape[1]),
                                    dtype=gcn_penult.dtype, device=self.device)
            token_gnn[valid_ln] = gcn_penult[local_node_ids[valid_ln]]
            ret["h_mutual"] = h_mutual
            ret["token_gnn"] = token_gnn
            ret["token_node_ids"] = batch_node_ids
        return ret

    def _mutual_active(self):
        """Mutual learning is on only when a mode is set and at least one
        direction has non-zero weight (both zero -> skip pass-1 entirely, so
        training is bit-exact with the baseline, RNG included)."""
        return (getattr(self, "mutual_mode", "off") != "off"
                and self.lambda_llm_to_gnn + self.lambda_gnn_to_llm > 0)

    def _refine_adjacency(self, sub_adj, sub_node_ids, node_map, token_node_ids,
                          h_mutual, ctx_penult):
        """Semantic Graph Refinement (scheme 2).

        The LLM proposes candidate semantic edges between the batch's items:
            s_sem(i,j) = cos(h_i, h_j)   (mid-layer semantic similarity)
        A learned gate decides acceptance ("LLM proposes, graph decides"):
            gamma_ij = sigmoid(g([h_i+h_j; z_i+z_j]))   (h detached in feat)
            w_ij     = gamma_ij * relu(s_sem(i,j))
        The gated block is added to the subgraph adjacency; NGCF's internal
        degree renormalization then lets the *graph* decide how much each
        semantic edge actually participates in message passing. Gradients
        through the edge weights flow back into the LLM (semantic -> graph
        guidance) and through the gate into the final objective.

        Returns the refined sparse adjacency (input unchanged if < 2 items).
        """
        item_mask = token_node_ids >= self.num_users
        item_token_rows = item_mask.nonzero(as_tuple=True)[0]
        if item_token_rows.numel() < 2:
            return sub_adj
        item_global = token_node_ids[item_token_rows]
        local = torch.tensor([node_map.get(int(g), -1) for g in item_global],
                             device=sub_adj.device, dtype=torch.long)
        valid = local >= 0
        if int(valid.sum()) < 2:
            return sub_adj
        local = local[valid]
        h_items = h_mutual[item_token_rows[valid]]        # (K, hidden), grad -> LLM
        z_items = ctx_penult[local]                       # (K, d), plain-graph context
        K = local.numel()
        # symmetric pair features for the gate (h detached: the gate learns
        # acceptance, it does not backprop into the LLM representations)
        with torch.no_grad():
            h_sum = h_items.unsqueeze(1) + h_items.unsqueeze(0)     # (K,K,H)
            z_sum = z_items.unsqueeze(1) + z_items.unsqueeze(0)     # (K,K,d)
        feat = torch.cat([h_sum.reshape(K * K, -1), z_sum.reshape(K * K, -1)], dim=-1)
        gamma = torch.sigmoid(self.sem_gate(feat)).reshape(K, K)
        s_sem = F.cosine_similarity(h_items.unsqueeze(1), h_items.unsqueeze(0), dim=-1)
        w = gamma * F.relu(s_sem)
        w = w - torch.diag(w.diagonal())                            # no self-edges
        dense = sub_adj.to_dense()
        dense[local.unsqueeze(1), local.unsqueeze(0)] = \
            dense[local.unsqueeze(1), local.unsqueeze(0)] + w.to(dense.dtype)
        return dense.to_sparse()

    def _make_injection_hook(self, slot):
        """Forward hook injecting the collaborative expert at decoder layer
        `self.injection_layers[slot]`:

            h' = h + alpha_l * rho_token * B_G_l(P_G_l(z_token))

        B_G is zero-init (warm-start safe); rho is a per-token router on
        [h_detached; z]; alpha_l comes from the B10 layer router. The hook is
        armed only during the main forward (self._injection_on), so the mutual
        pass-1 never injects.
        """
        def hook(module, inputs, output):
            if not self._injection_on or self._inj is None:
                return
            inj = self._inj
            bi, tp, lni = inj["bi"], inj["tp"], inj["lni"]
            if bi is None or bi.numel() == 0:
                return
            hs = output[0] if isinstance(output, tuple) else output
            valid = lni >= 0
            if not bool(valid.any()):
                return
            with torch.cuda.amp.autocast(enabled=False):
                z = inj["tg"][valid]                                   # (N,256) fp32
                e = self.collab_A[slot](z)
                dh = self.collab_B[slot](e)                            # (N,hidden)
                h = hs[bi[valid], tp[valid]].float()
                rho = torch.sigmoid(self.inj_router[slot](torch.cat([h.detach(), z], dim=-1)))
                # relative-norm injection: fixed fraction of the token hidden
                # norm, so the add survives fp16 quantization at every layer
                dh_n = dh.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                h_n = h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                add = inj["a_tok"][valid, slot].unsqueeze(-1) * rho \
                    * self.expert_scale * (dh / dh_n) * h_n
            hs_new = hs.clone()
            hs_new[bi[valid], tp[valid]] = hs_new[bi[valid], tp[valid]] + add.to(hs.dtype)
            if isinstance(output, tuple):
                return (hs_new,) + tuple(output[1:])
            return hs_new
        return hook

    def compute_mutual_losses(self, fb, yes_logit=None, labels=None):
        """Bidirectional mutual learning between the LLM semantic view and the
        GNN collaborative view. Neither side is frozen: each loss detaches only
        its target side, so the other side receives gradients, and LoRA keeps
        updating through L_rec and L_{G->L} regardless.

        B3: batch-wide relation alignment. Every user token in the batch is
        related against every candidate item token in the batch; only the
        relation *ranking* is aligned, never representation coordinates.
        Both directions are computed per user row (M,) so B11 can gate
        knowledge transfer per sample.

        B2: in-batch InfoNCE between projector-mapped views of the same ID
        token entity (gating not supported).

        B11 gating (mutual_gate != "off", rank mode only): knowledge transfer
        happens per sample only when one side is confident and the other is
        not —
            m_LG = 1[q_L > tau_h and q_G < tau_l]  (LLM teaches GNN)
            m_GL = 1[q_G > tau_h and q_L < tau_l]  (GNN teaches LLM)
        or, in "disagreement" mode, when |q_L - q_G| > gate_delta. A small GNN
        confidence head s_G = kappa * cos(z_u, z_target) (kappa learned via its
        own BCE against the label) supplies the collaborative side's opinion.

        REL mode (mutual_mode == "rel", CLLM4Rec-style semantic-collaborative
        cooperation): item-item relation matrix alignment over the batch's item
        tokens —
            S_L = cos(h_i, h_j)  (LLM mid-layer semantic relations)
            S_G = cos(z_i, z_j)  (GNN collaborative relations)
            L_S2G = mean_i w_i * (S_G_ij - sg(S_L_ij))^2,  w_i = q_L(sample(i))
        the teacher side is stop-gradiented per direction, and the per-item
        confidence weight makes the semantic signal proportional to how sure
        the LLM is (adaptive mutual cooperation, CLLM4Rec's lambda_c insight).

        Returns (loss_llm_to_gnn, loss_gnn_to_llm, loss_conf or None).
        """
        h_mutual = fb["h_mutual"]        # (N, hidden), fp32, grad -> LLM side
        token_gnn = fb["token_gnn"]      # (N, d), fp32, grad -> GNN side
        token_node_ids = fb["token_node_ids"]
        batch_indices = fb["batch_indices"]

        with torch.cuda.amp.autocast(enabled=False):
            num_users = self.num_users
            user_mask = token_node_ids < num_users
            user_rows = user_mask.nonzero(as_tuple=True)[0]
            # sample-level GNN confidence via the kappa head (shared by B11
            # gating, REL weighting, and the L_GNN^rec-style calibration loss)
            z_u = token_gnn[user_rows]                     # (S,256), sample order
            tgt_rows = []
            for b in batch_indices.unique():
                rows = (batch_indices == b).nonzero(as_tuple=True)[0]
                item_rows_s = rows[~user_mask[rows]]
                tgt_rows.append(item_rows_s[-1] if item_rows_s.numel() > 0 else rows[0])
            tgt_rows = torch.stack(tgt_rows)
            z_t = token_gnn[tgt_rows]
            kap = self.gnn_conf_kappa if hasattr(self, "gnn_conf_kappa") else 5.0
            s_g_sample = kap * F.cosine_similarity(z_u, z_t, dim=-1)
            q_g_sample = torch.sigmoid(s_g_sample).detach()
            q_l_sample = torch.sigmoid(yes_logit.float().reshape(-1)).detach() \
                if yes_logit is not None else None
            loss_conf = None
            if labels is not None and hasattr(self, "gnn_conf_kappa"):
                loss_conf = F.binary_cross_entropy_with_logits(
                    s_g_sample, labels.to(s_g_sample.dtype).reshape(-1))

            if self.mutual_mode == "b3":
                if self.mutual_rank_loss:
                    loss_l2g_u, loss_g2l_u = self._b3_ranking_losses(
                        h_mutual, token_gnn, token_node_ids, batch_indices)
                else:
                    loss_l2g_u, loss_g2l_u = self._b3_losses(h_mutual, token_gnn,
                                                             token_node_ids, batch_indices)
                gate_on = (self.mutual_gate != "off"
                           and yes_logit is not None and labels is not None)
                if gate_on:
                    q_l = q_l_sample
                    q_g = q_g_sample
                    if self.mutual_gate == "threshold":
                        m_lg = ((q_l > self.gate_tau_high) & (q_g < self.gate_tau_low)).float()
                        m_gl = ((q_g > self.gate_tau_high) & (q_l < self.gate_tau_low)).float()
                    else:  # disagreement
                        m = ((q_l - q_g).abs() > self.gate_delta).float()
                        m_lg, m_gl = m, m
                else:
                    m_lg = torch.ones_like(s_g_sample)
                    m_gl = torch.ones_like(s_g_sample)
                denom_lg = m_lg.sum().clamp(min=1.0)
                denom_gl = m_gl.sum().clamp(min=1.0)
                loss_l2g = (m_lg * loss_l2g_u).sum() / denom_lg
                loss_g2l = (m_gl * loss_g2l_u).sum() / denom_gl
                self._last_gate = {"mLG": m_lg.mean().item(), "mGL": m_gl.mean().item(),
                                   "qG": q_g_sample.mean().item()}
            elif self.mutual_mode == "rlm":
                # RLMRec-Con: bridge-space cross-view contrastive alignment
                item_rows = (~user_mask).nonzero(as_tuple=True)[0]
                h_items = h_mutual[item_rows]                     # (K, hidden)
                z_g_items = token_gnn[item_rows]                  # (K, d)
                z_l = self.rlm_head_L(h_items)                    # (K, d_b) grad -> LLM
                z_g = self.rlm_head_G(z_g_items)                  # (K, d_b) grad -> GNN
                z_l_n = F.normalize(z_l, dim=-1)
                z_g_n = F.normalize(z_g, dim=-1)
                tgt = torch.arange(z_l.shape[0], device=z_l.device)
                # L->G: updates GNN + P_G (z_L teacher, stop-grad)
                loss_l2g = F.cross_entropy(z_g_n @ z_l_n.detach().t() / self.rlm_tau, tgt)
                # G->L: updates LLM + P_L (z_G teacher, stop-grad)
                loss_g2l = F.cross_entropy(z_l_n @ z_g_n.detach().t() / self.rlm_tau, tgt)
            elif self.mutual_mode == "rel":
                item_rows = (~user_mask).nonzero(as_tuple=True)[0]
                q_l_item = q_l_sample[batch_indices[item_rows]] if q_l_sample is not None else None
                q_g_item = q_g_sample[batch_indices[item_rows]]
                # EMA momentum teacher: chase slowly-moving per-item semantic
                # memories instead of the fast-moving online hidden states
                s_l_teacher = None
                if self.rel_ema_momentum > 0 and hasattr(self, "_sem_bank"):
                    with torch.no_grad():
                        item_ids = token_node_ids[item_rows]
                        h_new = h_mutual[item_rows].detach()
                        bank_rows = self._sem_bank[item_ids]
                        init_mask = bank_rows.abs().sum(-1) == 0
                        h_teach = torch.where(init_mask.unsqueeze(-1), h_new, bank_rows)
                        s_l_teacher = F.cosine_similarity(
                            h_teach.unsqueeze(1), h_teach.unsqueeze(0), dim=-1)
                        m_e = self.rel_ema_momentum
                        self._sem_bank[item_ids] = m_e * bank_rows + (1 - m_e) * h_new
                loss_l2g, loss_g2l = self._rel_losses(h_mutual, token_gnn, item_rows,
                                                      w_l_item=q_l_item, w_g_item=q_g_item,
                                                      s_l_teacher=s_l_teacher)
                self._last_gate = {"qL": (q_l_item.mean().item() if q_l_item is not None else -1),
                                   "qG": q_g_item.mean().item()}
            else:  # b2
                q_l = F.normalize(self.mutual_proj_L(h_mutual), dim=-1)
                q_g = F.normalize(self.mutual_proj_G(token_gnn), dim=-1)
                n = q_l.shape[0]
                tgt = torch.arange(n, device=q_l.device)
                # LLM -> GNN: q_G queries against detached q_L keys -> GNN learns
                loss_l2g = F.cross_entropy(q_g @ q_l.detach().t() / self.b2_temp,
                                           tgt).reshape(1)
                # GNN -> LLM: q_L queries against detached q_G keys -> LLM learns
                loss_g2l = F.cross_entropy(q_l @ q_g.detach().t() / self.b2_temp,
                                           tgt).reshape(1)

        # optional iteration-based warmup: full weight after mutual_warmup_iters
        self._mutual_step = getattr(self, "_mutual_step", 0) + 1
        if self.mutual_warmup_iters > 0:
            ramp = min(1.0, self._mutual_step / float(self.mutual_warmup_iters))
            loss_l2g = loss_l2g * ramp
            loss_g2l = loss_g2l * ramp

        st = self._mutual_stats
        st["n"] += 1
        st["l2g"] += loss_l2g.item()
        st["g2l"] += loss_g2l.item()
        if st["n"] % 100 == 0:
            extra = ""
            if self.mutual_mode in ("b3", "rel") and hasattr(self, "_last_gate"):
                g = self._last_gate
                parts = [f"{k}={v:.3f}" for k, v in g.items()]
                if self.mutual_mode == "b3" and hasattr(self, "gnn_conf_kappa"):
                    parts.append(f"kappa={float(self.gnn_conf_kappa):.2f}")
                extra = " " + " ".join(parts)
            print(f"[mutual:{self.mutual_mode}] iters={st['n']} "
                  f"L_L->G={st['l2g'] / st['n']:.4f} L_G->L={st['g2l'] / st['n']:.4f}"
                  + extra, flush=True)
        return loss_l2g, loss_g2l, loss_conf

    def _b3_losses(self, h_mutual, token_gnn, token_node_ids, batch_indices):
        """Per-user KL vectors (M,) so B11 can gate per sample."""
        num_users = self.num_users
        user_mask = token_node_ids < num_users
        item_mask = ~user_mask
        if user_mask.sum() == 0 or item_mask.sum() < 2:
            # degenerate batch (no user token, or a single candidate -> the
            # relation distribution is a point mass and carries no signal)
            return h_mutual.sum() * 0.0 * user_mask.float(), \
                h_mutual.sum() * 0.0 * user_mask.float()
        # relation matrices: rows = users (M), cols = candidate items (K)
        h_u, h_i = h_mutual[user_mask], h_mutual[item_mask]
        z_u, z_i = token_gnn[user_mask], token_gnn[item_mask]
        # LLM relation distribution
        r_l = F.cosine_similarity(h_u.unsqueeze(1), h_i.unsqueeze(0), dim=-1) \
            / self.rel_temp_llm                       # (M, K)
        logp_l = F.log_softmax(r_l, dim=-1)
        p_l = logp_l.exp()
        # GNN relation distribution
        r_g = F.cosine_similarity(z_u.unsqueeze(1), z_i.unsqueeze(0), dim=-1) \
            / self.rel_temp_gnn                       # (M, K)
        logp_g = F.log_softmax(r_g, dim=-1)
        p_g = logp_g.exp()
        # symmetric KL with the target side detached per direction, per user row
        target_l = p_l.detach()
        loss_l2g_u = (target_l * (torch.log(target_l.clamp_min(1e-12)) - logp_g)).sum(dim=-1)
        target_g = p_g.detach()
        loss_g2l_u = (target_g * (torch.log(target_g.clamp_min(1e-12)) - logp_l)).sum(dim=-1)
        return loss_l2g_u, loss_g2l_u

    def _b3_ranking_losses(self, h_mutual, token_gnn, token_node_ids, batch_indices):
        """Scale-free relational *ranking* match (fixes the KL over-alignment).

        KL(softmax) matching forces the *absolute confidence* of the two views
        to coincide, so once the candidate orderings agree the only way to push
        KL lower is to warp one space's cosine geometry into the other's —
        that is what degraded valid AUC after epoch 1 in the KL run (train loss
        kept improving on the train graph while the warped representations
        stopped transferring to the valid graph). Here each direction only
        transfers the teacher's item *ordering*: the teacher (detached)
        contributes the desired sign of each candidate-pair gap and a weight
        proportional to how decisive it is (near-ties exert no pressure); the
        student pays a hinge penalty until its own gap realizes that order
        beyond rank_margin. The loss is exactly zero once both views agree
        with margin, so it cannot keep grinding the representations apart.
        """
        num_users = self.num_users
        user_mask = token_node_ids < num_users
        item_mask = ~user_mask
        if user_mask.sum() == 0 or item_mask.sum() < 2:
            zero = h_mutual.sum() * 0.0 * user_mask.float()
            return zero, zero.clone()
        h_u, h_i = h_mutual[user_mask], h_mutual[item_mask]
        z_u, z_i = token_gnn[user_mask], token_gnn[item_mask]
        r_l = F.cosine_similarity(h_u.unsqueeze(1), h_i.unsqueeze(0), dim=-1)   # (M, K)
        r_g = F.cosine_similarity(z_u.unsqueeze(1), z_i.unsqueeze(0), dim=-1)   # (M, K)
        # all candidate pairs per user row
        ii, jj = torch.triu_indices(r_l.shape[1], r_l.shape[1], offset=1,
                                    device=r_l.device)
        d_l = r_l[:, ii] - r_l[:, jj]   # (M, P) LLM-view pair gaps
        d_g = r_g[:, ii] - r_g[:, jj]   # (M, P) GNN-view pair gaps
        m, w_ref = self.rank_margin, self.rank_weight_ref
        # LLM -> GNN: LLM is the teacher (detached sign + decisiveness weight)
        w_l = (d_l.detach().abs() / w_ref).clamp(max=1.0)
        loss_l2g_u = (w_l * F.relu(m - torch.sign(d_l.detach()) * d_g)).mean(dim=-1)  # (M,)
        # GNN -> LLM: GNN is the teacher
        w_g = (d_g.detach().abs() / w_ref).clamp(max=1.0)
        loss_g2l_u = (w_g * F.relu(m - torch.sign(d_g.detach()) * d_l)).mean(dim=-1)  # (M,)
        return loss_l2g_u, loss_g2l_u

    def _rel_losses(self, h_mutual, token_gnn, item_rows, w_l_item=None, w_g_item=None,
                    s_l_teacher=None):
        """REL mode: item-item relation matrix alignment (CLLM4Rec-inspired).

        The two views share their *structure*, never coordinates:
            S_L = cos(h_i, h_j)  over the batch's item tokens — which items the
                                 LLM considers semantically alike
            S_G = cos(z_i, z_j)  over the GNN penultimate embeddings — which
                                 items the interaction graph holds together
        L_S2G teaches the GNN the semantic structure (e.g. two sci-fi movies
        with disjoint interaction neighborhoods should still be related);
        L_G2L (optional, second step) teaches the LLM the collaborative
        structure. The teacher similarity matrix is stop-gradiented per
        direction, and per-item confidence weights (q_L / q_G of the item's
        sample) realize CLLM4Rec's insight that the mutual regularization
        strength must adapt to how trustworthy each side currently is.

        s_l_teacher (optional): EMA momentum teacher — when provided, the S->G
        direction chases the slowly-moving bank relations instead of the online
        ones. rel_topk > 0 restricts the S->G MSE to the teacher's TopK local
        neighborhood per item (manifold alignment, self-pairs excluded).

        Returns (loss_s2g, loss_g2l), fp32 scalars.
        """
        K = int(item_rows.numel())
        if K < 2:
            zero = h_mutual.sum() * 0.0
            return zero, zero.clone()
        h_i = h_mutual[item_rows]            # (K, hidden), grad -> LLM
        z_i = token_gnn[item_rows]           # (K, d), grad -> GNN
        s_l = F.cosine_similarity(h_i.unsqueeze(1), h_i.unsqueeze(0), dim=-1)  # (K,K)
        s_g = F.cosine_similarity(z_i.unsqueeze(1), z_i.unsqueeze(0), dim=-1)  # (K,K)
        s_l_t = s_l_teacher if s_l_teacher is not None else s_l
        # S -> G: GNN's collaborative relations move toward the (detached)
        # semantic relations; weighted by the LLM's per-sample confidence
        diff = (s_g - s_l_t.detach()) ** 2
        if self.rel_topk and 0 < self.rel_topk < K:
            teacher = s_l_t.detach().clone()
            teacher.fill_diagonal_(float("-inf"))            # exclude self-pairs
            topk_idx = teacher.topk(self.rel_topk, dim=-1).indices
            mask = torch.zeros_like(diff, dtype=torch.bool).scatter_(1, topk_idx, True)
            diff = diff * mask
            if w_l_item is not None:
                w = w_l_item.clamp_min(0.0).unsqueeze(-1)
                loss_s2g = (w * diff).sum() / (w.sum().clamp_min(1e-6) * self.rel_topk)
            else:
                loss_s2g = diff.sum() / mask.sum()
        else:
            if w_l_item is not None:
                w = w_l_item.clamp_min(0.0)
                loss_s2g = (w * diff.mean(dim=1)).sum() / w.sum().clamp_min(1e-6)
            else:
                loss_s2g = diff.mean()
        # G -> L: the LLM's semantic relations move toward the (detached)
        # collaborative relations; weighted by the GNN's per-sample confidence
        if w_g_item is not None:
            w = w_g_item.clamp_min(0.0)
            loss_g2l = (w * ((s_l - s_g.detach()) ** 2).mean(dim=1)).sum() / w.sum().clamp_min(1e-6)
        else:
            loss_g2l = ((s_l - s_g.detach()) ** 2).mean()
        return loss_s2g, loss_g2l

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
        user_rows = user_mask.nonzero(as_tuple=True)[0]           # in batch-row order
        tgt_rows = []
        for r in batch_indices.unique():
            rows_r = (batch_indices == r).nonzero(as_tuple=True)[0]
            item_rows_r = rows_r[~user_mask[rows_r]]
            tgt_rows.append(item_rows_r[-1] if item_rows_r.numel() > 0 else rows_r[0])
        tgt_rows = torch.stack(tgt_rows)
        kap = self.gnn_conf_kappa if hasattr(self, "gnn_conf_kappa") else 5.0
        return kap * F.cosine_similarity(token_gnn[user_rows], token_gnn[tgt_rows], dim=-1)

    def _pref_distill_loss(self, fb, logits, samples):
        """Preference-level G->L distillation (candidate groups).

        For each candidate group (one user/history prompt with K candidate
        targets), the GNN's preference distribution
            P_G = softmax(kappa*cos(z_u,z_c)/tau_G)      (teacher, stop-grad)
        is distilled into the LLM's Yes-probability distribution
            P_L = softmax(yes_logit/tau_L)               (student)
        weighted by the GNN's confidence. This is the mechanism by which the
        collaborative space *raises the Yes probability* of the items it
        considers better matches — preference level, not feature level.
        """
        token_gnn = fb["token_gnn"]
        s_g_row = self._row_gnn_scores(fb)                       # (B,)
        q_g_row = torch.sigmoid(s_g_row).detach()
        groups = samples["cand_group"]
        loss_sum = logits.sum() * 0.0
        w_sum = torch.zeros((), device=logits.device)
        with torch.cuda.amp.autocast(enabled=False):
            for g in groups.unique():
                rows_g = (groups == g).nonzero(as_tuple=True)[0]
                if rows_g.numel() < 2:
                    continue
                p_l = F.log_softmax(logits[rows_g].float() / self.pref_tau_l, dim=-1)
                p_g = F.softmax(s_g_row[rows_g] / self.pref_tau_g, dim=-1).detach()
                w = q_g_row[rows_g].mean()
                kl = (p_g * (torch.log(p_g.clamp_min(1e-12)) - p_l)).sum()
                loss_sum = loss_sum + w * kl
                w_sum = w_sum + w
        return loss_sum / w_sum.clamp_min(1e-6)

    def _pref_mutual_loss(self, fb, logits, samples):
        """TCA-style prediction-level semantic-collaborative mutual
        optimization over candidate groups (scheme: prediction-level coupling).

        For each candidate group (one user prompt, K candidates):

        G->L (soft target, updates LLM only):
            p_G    = softmax(s_G/tau_G)                    (collaborative pref)
            ỹ(c)   = (1-alpha_u)*onehot(label) + alpha_u*sg(p_G(c))
            L_soft = -sum_c ỹ(c)*log softmax(yes_c/tau_L)
        unlike a fixed KL target, the gradient of this soft-CE depends on both
        the soft target AND the LLM's current prediction (TCA4Rec's point).

        L->G (semantic supervision, updates GNN/kappa only):
            p_L    = softmax(yes_c/tau_L)                  (teacher, stop-grad)
            L_L2G  = beta_u * KL(sg(p_L) || p_G)

        Confidence gates (CLLM4Rec's lambda_c insight): a u = alpha_max*c_G,
        beta_u = beta_max*c_L with c = 1 - H(p)/log K (entropy confidence);
        both are warmed up linearly over tca_warmup_iters.

        Returns (loss_soft, loss_l2g).
        """
        s_g_row = self._row_gnn_scores(fb)                       # (B,) GNN scores
        groups = samples["cand_group"]
        labels = samples["label"].float()                        # 1=target, 0=neg
        self._tca_step = getattr(self, "_tca_step", 0) + 1
        warm = min(1.0, self._tca_step / float(max(self.tca_warmup_iters, 1))) \
            if self.tca_warmup_iters > 0 else 1.0
        soft_sum = logits.sum() * 0.0
        l2g_sum = logits.sum() * 0.0
        gcnt = 0
        with torch.cuda.amp.autocast(enabled=False):
            for g in groups.unique():
                rows = (groups == g).nonzero(as_tuple=True)[0]
                if rows.numel() < 2:
                    continue
                Kc = rows.numel()
                y = labels[rows]
                logp_l = F.log_softmax(logits[rows].float() / self.pref_tau_l, dim=-1)
                p_l_live = logp_l.exp()
                p_l_det = p_l_live.detach()
                p_g = F.softmax(s_g_row[rows].float() / self.pref_tau_g, dim=-1).detach()
                # entropy-based confidence gates
                log_k = torch.log(torch.tensor(float(Kc), device=logp_l.device))
                c_g = 1.0 - (-(p_g * p_g.clamp_min(1e-12).log()).sum()) / log_k
                c_l = 1.0 - (-(p_l_det * p_l_det.clamp_min(1e-12).log()).sum()) / log_k
                alpha_u = self.tca_alpha_max * c_g * warm
                beta_u = self.tca_beta_max * c_l * warm
                # G->L: mixed soft target -> LLM
                y_tilde = (1.0 - alpha_u) * y + alpha_u * p_g
                soft_sum = soft_sum + (-(y_tilde * logp_l).sum())
                # L->G: semantic supervision -> GNN/kappa (teacher stop-grad)
                l2g_sum = l2g_sum + beta_u * (
                    p_l_det * (p_l_det.clamp_min(1e-12).log()
                               - F.log_softmax(s_g_row[rows].float() / self.pref_tau_g, dim=-1))).sum()
                gcnt += 1
        n = max(gcnt, 1)
        return soft_sum / n, l2g_sum / n

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

    def forward_v2(self, samples):
        user_selective_prompts = False
        if hasattr(samples, 'question_split'):
            print('VQA Batch')
            raise NotImplementedError("not implement")
        elif self.prompt_list:
            prompt = random.choice(self.prompt_with_p([5, 5, 5, 1]))
            # print("Selected prompt:", prompt)
            sample_embeds, atts_samples, samples_encode = self.prompt_based_encode_v2(prompt, samples)
            # print("sample_embeds shape:", sample_embeds.shape)
            # print("atts_samples shape:", atts_samples.shape)
            # print("samples_encode keys:", samples_encode.keys())

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
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
            -100)
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
            fb = self.compute_feedback_gcn(samples, samples_encode,
                                           samples_encode['replaced_idx'],
                                           inputs_embeds, attention_mask)
            if self.injection_mode == "legacy":
                gcn_output = fb['gcn_output']
                # original train-path post-processing (kept bit-exact with the old code)
                gcn_output = torch.nn.functional.normalize(gcn_output, dim=-1) * 0.1
                gcn_output = gcn_output.to(dtype=torch.float16)

                batch_indices = fb['batch_indices']
                token_positions = fb['token_positions']
                local_node_ids = fb['local_node_ids']

                if self.use_lora:
                    for my_linear in self.my_linears:
                        my_linear.gcn_output = gcn_output
                        my_linear.batch_indices = batch_indices
                        my_linear.token_positions = token_positions
                        my_linear.local_node_ids = local_node_ids
            elif self.use_lora and self.injection_mode == "expert":  # B9/B10
                user_rows = (fb['token_node_ids'] < self.num_users).nonzero(as_tuple=True)[0]
                z_u = fb['token_gnn'][user_rows].float()
                alpha = torch.softmax(self.layer_router(z_u), dim=-1)     # (S, L)
                self._last_alpha = alpha
                self._inj = {"bi": fb['batch_indices'],
                             "tp": fb['token_positions'],
                             "lni": fb['local_node_ids'],
                             "tg": fb['token_gnn'],
                             "a_tok": alpha[fb['batch_indices']]}
                self._injection_on = True

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

            if self.use_lora and self.injection_mode == "legacy":
                for my_linear in self.my_linears:
                    my_linear.gcn_output = None
                    my_linear.batch_indices = None
                    my_linear.token_positions = None
                    my_linear.local_node_ids = None
            elif self.use_lora and self.injection_mode == "expert":
                self._injection_on = False
                self._inj = None
            elif self.use_lora and self.injection_mode == "cflora":
                for my_linear in self.my_linears:
                    my_linear.cf_m = None

        pos_ans_id = self.llama_tokenizer(ans_[int(1)], add_special_tokens=False).input_ids[0]
        neg_ans_id = self.llama_tokenizer(ans_[int(0)], add_special_tokens=False).input_ids[0]
        logits = outputs.logits[:, -t_posi, :][:, pos_ans_id]
        loss_rec = nn.functional.binary_cross_entropy_with_logits(logits, samples['label'].float())
        loss = loss_rec

        ret = {"loss": loss}
        if self.training:
            diag = {"loss_rec": loss_rec.detach()}
            if self._mutual_active():
                loss_l2g, loss_g2l, loss_conf = self.compute_mutual_losses(
                    fb, logits.detach(), samples['label'].float())
                loss = loss + self.lambda_llm_to_gnn * loss_l2g + self.lambda_gnn_to_llm * loss_g2l
                if loss_conf is not None:
                    loss = loss + self.gnn_conf_loss_weight * loss_conf
                diag.update({"loss_l2g": loss_l2g.detach(), "loss_g2l": loss_g2l.detach()})
                if loss_conf is not None:
                    diag["loss_conf"] = loss_conf.detach()
                if self.injection_mode == "expert" and self.router_entropy_weight > 0 \
                        and hasattr(self, "_last_alpha"):
                    a = self._last_alpha
                    ent = -(a * (a + 1e-12).log()).sum(-1).mean()
                    loss = loss + self.router_entropy_weight * ent
            if self.pref_distill_weight > 0 and "cand_group" in samples:
                if self.pref_tca:
                    loss_soft, loss_l2g_pref = self._pref_mutual_loss(fb, logits, samples)
                    print(f"[tca-dbg] soft={loss_soft.item():.4f} l2g={loss_l2g_pref.item():.4f} "
                          f"cand={int(samples['cand_group'].max())+1} groups", flush=True)
                    loss = loss + self.pref_distill_weight * (loss_soft + loss_l2g_pref)
                    diag["loss_soft"] = loss_soft.detach()
                    diag["loss_pref"] = loss_l2g_pref.detach()
                else:
                    loss_pref = self._pref_distill_loss(fb, logits, samples)
                    loss = loss + self.pref_distill_weight * loss_pref
                    diag["loss_pref"] = loss_pref.detach()
            diag["logits"] = logits.detach()
            # CRITICAL: ret["loss"] was bound to the pre-auxiliary loss tensor;
            # rebind it so the runner's backward covers rec + ALL aux losses
            ret["loss"] = loss
            # ---- gradient-aware cooperation (PCGrad-style) ----
            # project out the component of the auxiliary gradient that directly
            # opposes the main L_rec gradient on shared parameters, so mutual
            # learning can never step AGAINST the primary LLM objective
            if self.grad_projection and loss.requires_grad:
                params = [p for p in self.parameters() if p.requires_grad]
                g_main = torch.autograd.grad(loss_rec, params,
                                             retain_graph=True, allow_unused=True)
                g_aux = torch.autograd.grad(loss - loss_rec, params,
                                            retain_graph=True, allow_unused=True)
                with torch.no_grad():
                    for p, gm, ga in zip(params, g_main, g_aux):
                        if ga is None:
                            g = gm
                        elif gm is None:
                            g = ga
                        else:
                            dot = (ga * gm).sum()
                            if dot < 0:
                                gn2 = (gm * gm).sum().clamp_min(1e-12)
                                ga = ga - (dot / gn2) * gm
                            g = gm + ga
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad = p.grad + g
                # the runner's own backward must contribute exactly nothing now
                trigger = sum((p * 0.0).sum() for p in params)
                ret["loss"] = loss.detach() + trigger
            ret.update(diag)
            # guard: the runner calls loss.backward(), so the returned "loss"
            # must carry a grad_fn even when diag carries detached copies
            if "loss" not in ret or not ret["loss"].requires_grad:
                ret["loss"] = loss
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
        # Load adjacency matrix based on sample mode
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
            sample_embeds, atts_samples, replaced_idx = self.recprompt_wrap_v2(samples_encode, samples, atts_samples,
                                                                               prompt)

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
        empty_targets = torch.ones([atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(
            -100)
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
                fb = self.compute_feedback_gcn(samples, samples_encode, replaced_idx,
                                               inputs_embeds, attention_mask)
                if self.injection_mode == "legacy":
                    gcn_output = fb['gcn_output']
                    # original eval-path post-processing (no extra normalize here)
                    gcn_output = gcn_output.to(dtype=torch.float16)

                    batch_indices = fb['batch_indices']
                    token_positions = fb['token_positions']
                    local_node_ids = fb['local_node_ids']
                    # Set GCN data
                    for my_linear in self.my_linears:
                        my_linear.gcn_output = gcn_output
                        my_linear.batch_indices = batch_indices
                        my_linear.token_positions = token_positions
                        my_linear.local_node_ids = local_node_ids
                elif self.injection_mode == "cflora":  # CFLoRA eval: cf_m set in compute_feedback_gcn
                    self._cflora_eval_pending = True
                elif self.injection_mode == "expert":  # B9/B10 eval injection
                    user_rows = (fb['token_node_ids'] < self.num_users).nonzero(as_tuple=True)[0]
                    z_u = fb['token_gnn'][user_rows].float()
                    alpha = torch.softmax(self.layer_router(z_u), dim=-1)
                    self._inj = {"bi": fb['batch_indices'],
                                 "tp": fb['token_positions'],
                                 "lni": fb['local_node_ids'],
                                 "tg": fb['token_gnn'],
                                 "a_tok": alpha[fb['batch_indices']]}
                    self._injection_on = True
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
            # Clear GCN data
            if self.use_lora and self.injection_mode == "legacy":
                for my_linear in self.my_linears:
                    my_linear.gcn_output = None
                    my_linear.batch_indices = None
                    my_linear.token_positions = None
                    my_linear.local_node_ids = None
            elif self.use_lora and self.injection_mode == "expert":
                self._injection_on = False
                self._inj = None
            if getattr(self, "_cflora_eval_pending", False):
                for _ml in getattr(self, "my_linears", []):
                    _ml.cf_m = None
                self._cflora_eval_pending = False
            pos_ans_id = self.llama_tokenizer(pos_ans, add_special_tokens=False).input_ids[0]
            neg_ans_id = self.llama_tokenizer(neg_ans, add_special_tokens=False).input_ids[0]
            logits = outputs.logits[:, -t_posi, :][:, pos_ans_id]
            # optional inference-time collaborative prior: items the GNN considers
            # better matches get their Yes logit raised by beta * s_G
            if getattr(self, "eval_fuse_beta", 0.0) > 0 and self.use_lora and "token_gnn" in fb:
                s_g_row = self._row_gnn_scores(fb)
                logits = logits + self.eval_fuse_beta * s_g_row.to(logits.dtype)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, samples['label'].float())
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
        embedding_size = cfg.get("rec_emb_size")
        freeze_rec = cfg.get("freeze_rec", True)
        rec_precision = cfg.get("rec_precision", 'fp16')
        lora_config = cfg.get("lora_config")
        llama_model = cfg.get("llama_model")
        proj_token_num = cfg.get("proj_token_num")
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
        semantic_feedback = cfg.get("semantic_feedback", False)
        semantic_layer = cfg.get("semantic_layer", 8)
        feedback_detach = cfg.get("feedback_detach", False)
        feedback_gate_bias_init = cfg.get("feedback_gate_bias_init", -2.0)
        mutual_mode = cfg.get("mutual_mode", "off")
        mutual_layer = cfg.get("mutual_layer", 8)
        lambda_llm_to_gnn = cfg.get("lambda_llm_to_gnn", 0.005)
        lambda_gnn_to_llm = cfg.get("lambda_gnn_to_llm", 0.02)
        rel_temp_llm = cfg.get("rel_temp_llm", 1.0)
        rel_temp_gnn = cfg.get("rel_temp_gnn", 1.0)
        b2_temp = cfg.get("b2_temp", 0.1)
        mutual_rank_loss = cfg.get("mutual_rank_loss", False)
        rank_margin = cfg.get("rank_margin", 0.02)
        rank_weight_ref = cfg.get("rank_weight_ref", 0.05)
        mutual_warmup_iters = cfg.get("mutual_warmup_iters", 0)
        # normalize boolean variants (OmegaConf may parse "off"/on-like values
        # as booleans, which would trip the mutual_mode assert below)
        if isinstance(mutual_mode, bool):
            mutual_mode = "b3" if mutual_mode else "off"
        mutual_gate = cfg.get("mutual_gate", "off")
        gate_tau_high = cfg.get("gate_tau_high", 0.75)
        gate_tau_low = cfg.get("gate_tau_low", 0.6)
        gate_delta = cfg.get("gate_delta", 0.1)
        gnn_conf_kappa = cfg.get("gnn_conf_kappa", 5.0)
        gnn_conf_loss_weight = cfg.get("gnn_conf_loss_weight", 0.1)
        pref_distill_weight = cfg.get("pref_distill_weight", 0.0)
        pref_tau_l = cfg.get("pref_tau_l", 1.0)
        pref_tau_g = cfg.get("pref_tau_g", 1.0)
        eval_fuse_beta = cfg.get("eval_fuse_beta", 0.0)
        sem_refine = cfg.get("sem_refine", False)
        pref_tca = cfg.get("pref_tca", False)
        tca_alpha_max = cfg.get("tca_alpha_max", 0.2)
        tca_beta_max = cfg.get("tca_beta_max", 0.1)
        tca_warmup_iters = cfg.get("tca_warmup_iters", 500)
        rel_ema_momentum = cfg.get("rel_ema_momentum", 0.0)
        grad_projection = cfg.get("grad_projection", False)
        rel_topk = cfg.get("rel_topk", 0)
        # RLMRec-Con bridge-space configuration
        rlm_bridge_dim = cfg.get("rlm_bridge_dim", 256)
        rlm_tau = cfg.get("rlm_tau", 0.1)
        injection_mode = cfg.get("injection_mode", "legacy")
        injection_layers = cfg.get("injection_layers", None)
        router_entropy_weight = cfg.get("router_entropy_weight", 0.0)
        expert_init_scale = cfg.get("expert_init_scale", 0.02)
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
            freeze_lora=freeze_lora,
            freeze_proj=freeze_proj,
            lora_gnn=lora_gnn,
            lora_scale=lora_scale,
            gnn_scale=gnn_scale,
            gcn_layer=gcn_layer,
            dataset=dataset,
            semantic_feedback=semantic_feedback,
            semantic_layer=semantic_layer,
            feedback_detach=feedback_detach,
            feedback_gate_bias_init=feedback_gate_bias_init,
            mutual_mode=mutual_mode,
            mutual_layer=mutual_layer,
            lambda_llm_to_gnn=lambda_llm_to_gnn,
            lambda_gnn_to_llm=lambda_gnn_to_llm,
            rel_temp_llm=rel_temp_llm,
            rel_temp_gnn=rel_temp_gnn,
            b2_temp=b2_temp,
            mutual_rank_loss=mutual_rank_loss,
            rank_margin=rank_margin,
            rank_weight_ref=rank_weight_ref,
            mutual_warmup_iters=mutual_warmup_iters,
            mutual_gate=mutual_gate,
            gate_tau_high=gate_tau_high,
            gate_tau_low=gate_tau_low,
            gate_delta=gate_delta,
            gnn_conf_kappa=gnn_conf_kappa,
            gnn_conf_loss_weight=gnn_conf_loss_weight,
            pref_distill_weight=pref_distill_weight,
            pref_tau_l=pref_tau_l,
            pref_tau_g=pref_tau_g,
            eval_fuse_beta=eval_fuse_beta,
            rel_ema_momentum=rel_ema_momentum,
            grad_projection=grad_projection,
            rel_topk=rel_topk,
            sem_refine=sem_refine,
            rlm_bridge_dim=rlm_bridge_dim,
            rlm_tau=rlm_tau,
            pref_tca=pref_tca,
            tca_alpha_max=tca_alpha_max,
            tca_beta_max=tca_beta_max,
            tca_warmup_iters=tca_warmup_iters,
            injection_mode=injection_mode,
            injection_layers=injection_layers,
            router_entropy_weight=router_entropy_weight,
            expert_init_scale=expert_init_scale,
        )
        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load MiniGPT4Rec Checkpoint: {}".format(ckpt_path))
            # mmap: avoid loading the whole multi-GB checkpoint into RAM at once
            # (7B llama fp16 already resident; plain torch.load caused OOM-kill)
            ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print("loading message, msg.... ", msg)
            del ckpt  # release the mmap
            if os.path.exists(rec_config['pretrained_path']) and freeze_rec:
                model.rec_encoder.load_state_dict(torch.load(rec_config['pretrained_path'], map_location="cpu"))
        ans_type = cfg.get('ans_type')
        model.set_answer_type(mode=ans_type)
        model.print_prompt()
        return model