"""CoRA perceptual dynamic-weight generator plus controlled structural variants.

The "shared" mode preserves the released CoRA input path. Experimental modes
change how the same frozen MF user/item pair is represented or how the
generated rank basis is conditioned; they do not change training hyperparams.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.no31_cora_original.qformer import BertModel, BertConfig


class PerceptualWeightsGenerator(nn.Module):
    def __init__(
        self,
        num_queries,
        user_embedding_size,
        hidden_size,
        r,
        input_mode="concat",
        injection_mode="shared",
    ):
        super().__init__()
        self.embedding_size = user_embedding_size
        self.r = r
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        if input_mode not in ("concat", "two_tokens"):
            raise ValueError(f"Invalid CoRA input mode: {input_mode}")
        if injection_mode not in (
            "shared", "typed_tokens", "interaction_tokens",
            "layer_conditioned", "token_gate",
        ):
            raise ValueError(f"Invalid CoRA injection mode: {injection_mode}")
        self.input_mode = input_mode
        self.injection_mode = injection_mode
        self.qformer, self.query_tokens = self._init_qformer()
        self.final = nn.Linear(user_embedding_size * 2, hidden_size * r)
        nn.init.kaiming_uniform_(self.final.weight, a=math.sqrt(5))

        # Structural variant M1/M2: expose user and candidate as separate,
        # role-marked encoder tokens instead of one concatenated token.
        # The projection is shared so the ablation tests token structure,
        # not unrelated user/item projection capacity.
        if injection_mode in ("typed_tokens", "interaction_tokens"):
            width = user_embedding_size * 2
            self.entity_projection = nn.Linear(user_embedding_size, width)
            self.user_type = nn.Parameter(torch.empty(1, 1, width))
            self.item_type = nn.Parameter(torch.empty(1, 1, width))
            nn.init.normal_(self.user_type, mean=0.0, std=0.02)
            nn.init.normal_(self.item_type, mean=0.0, std=0.02)

        # Structural variant M2: an explicit user-item compatibility token.
        if injection_mode == "interaction_tokens":
            width = user_embedding_size * 2
            self.interaction_projection = nn.Linear(user_embedding_size, width)
            self.interaction_type = nn.Parameter(torch.empty(1, 1, width))
            nn.init.normal_(self.interaction_type, mean=0.0, std=0.02)

        # Structural variant M3: layer-specific rank-channel modulation is
        # conditioned on both the MF pair summary and the layer identity.
        # Small output initialization keeps the first forward near the
        # shared-CoRA equation without disabling gradients to the conditioner.
        if injection_mode == "layer_conditioned":
            width = user_embedding_size * 2
            self.layer_embedding = nn.Embedding(32, 32)
            self.layer_modulator = nn.Sequential(
                nn.Linear(width + 32, 128),
                nn.GELU(),
                nn.Linear(128, r),
            )
            nn.init.normal_(self.layer_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.layer_modulator[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.layer_modulator[-1].bias)

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

    def _encoder_tokens(self, ui_embedding):
        if ui_embedding.ndim != 2 or ui_embedding.size(-1) != 2 * self.embedding_size:
            raise RuntimeError("Unexpected CoRA CF input for tokenized mode")
        user, item = ui_embedding.split(self.embedding_size, dim=-1)
        user_token = self.entity_projection(user).unsqueeze(1) + self.user_type
        item_token = self.entity_projection(item).unsqueeze(1) + self.item_type
        if self.injection_mode == "typed_tokens":
            return torch.cat((user_token, item_token), dim=1)
        interaction = user * item
        interaction_token = (
            self.interaction_projection(interaction).unsqueeze(1)
            + self.interaction_type
        )
        return torch.cat((user_token, item_token, interaction_token), dim=1)

    def forward(self, ui_embedding, return_condition=False):
        if self.injection_mode in ("typed_tokens", "interaction_tokens"):
            encoder_states = self._encoder_tokens(ui_embedding)
        elif self.input_mode == "concat":
            # Exact released CoRA path: one 512-d encoder token.
            encoder_states = ui_embedding.unsqueeze(-2)
        else:
            # Historical two_tokens ablation: zero-padded user/item tokens.
            if ui_embedding.ndim != 2 or ui_embedding.size(-1) != 2 * self.embedding_size:
                raise RuntimeError("Unexpected CoRA CF input for two-token mode")
            user, item = ui_embedding.split(self.embedding_size, dim=-1)
            encoder_states = torch.stack(
                [
                    torch.cat([user, torch.zeros_like(item)], dim=-1),
                    torch.cat([torch.zeros_like(user), item], dim=-1),
                ],
                dim=1,
            )

        encoder_atts = torch.ones(
            encoder_states.size()[:-1], dtype=torch.long, device=encoder_states.device
        )
        query_output = self.qformer(
            query_embeds=self.query_tokens.expand(encoder_states.size(0), -1, -1),
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_atts,
            use_cache=True,
            return_dict=True,
        )
        # Keep the released projection/normalization equation for all modes.
        ui_weight = (
            F.normalize(
                self.final(query_output.last_hidden_state).mean(dim=1, keepdim=True),
                p=2,
                dim=-1,
            ).view(encoder_states.size(0), self.hidden_size, self.r)
        )
        if return_condition:
            condition = query_output.last_hidden_state.mean(dim=1)
            return ui_weight, condition
        return ui_weight
