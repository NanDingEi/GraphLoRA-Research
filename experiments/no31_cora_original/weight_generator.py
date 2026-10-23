"""Original CoRA PerceptualWeightsGenerator with a single batch-shape fix.

Source: VanillaCreamer/CoRA, minigpt4/models/ppeft/weight_generator.py.
Architecture, 8-layer 8-head Q-Former, 4 learnable queries, final linear,
mean-pooling, L2 normalization and full [batch, hidden, rank] matrix are
unchanged. In the publicly released source, query_tokens has batch=1 and
was passed without expansion; expanding across batch is needed for batch>1
cross-attention, including the GraphLoRA batch size of 3.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.no31_cora_original.qformer import BertModel, BertConfig


class PerceptualWeightsGenerator(nn.Module):
    def __init__(self, num_queries, user_embedding_size, hidden_size, r):
        super().__init__()
        self.embedding_size = user_embedding_size
        self.r = r
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.qformer, self.query_tokens = self._init_qformer()
        self.final = nn.Linear(user_embedding_size * 2, hidden_size * r)
        nn.init.kaiming_uniform_(self.final.weight, a=math.sqrt(5))

    def _init_qformer(self):
        config = BertConfig()
        config.add_cross_attention = True
        config.cross_attention_freq = 1
        config.query_length = self.num_queries
        config.num_hidden_layers = 8
        config.num_attention_heads = 8
        config.hidden_size = self.embedding_size * 2
        config.encoder_width = self.embedding_size * 2
        qformer = BertModel(config)
        query_tokens = nn.Parameter(
            torch.zeros(1, self.num_queries, self.embedding_size * 2)
        )
        query_tokens.data.normal_(mean=0.0, std=0.02)
        return qformer, query_tokens

    def forward(self, ui_embedding):
        ui_embedding = ui_embedding.unsqueeze(-2)
        ui_atts = torch.ones(
            ui_embedding.size()[:-1], dtype=torch.long, device=ui_embedding.device
        )
        query_output = self.qformer(
            # Original CoRA omitted expansion of batch=1 learned tokens.
            query_embeds=self.query_tokens.expand(ui_embedding.size(0), -1, -1),
            encoder_hidden_states=ui_embedding,
            encoder_attention_mask=ui_atts,
            use_cache=True,
            return_dict=True,
        )
        ui_weight = (
            F.normalize(
                self.final(query_output.last_hidden_state).mean(dim=1, keepdim=True),
                p=2,
                dim=-1,
            ).view(ui_embedding.size(0), self.hidden_size, self.r)
        )
        return ui_weight
