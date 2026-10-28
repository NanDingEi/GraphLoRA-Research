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
        rank_profile=None,
        group_projector_mode="off",
    ):
        super().__init__()
        self.embedding_size = user_embedding_size
        self.group_projector_mode = group_projector_mode
        if group_projector_mode not in (
            "off", "shared", "group2", "group4", "group8",
            "shared_group4", "shared_trunk4", "rank_group4", "group4_linear",
            "coarse_to_fine", "fine_to_coarse", "hierarchical_residual",
        ):
            raise ValueError(f"Unsupported group_projector_mode: {group_projector_mode}")
        if group_projector_mode != "off":
            if rank_profile is not None or injection_mode != "shared" or input_mode != "concat":
                raise ValueError("Group projectors require original rank16/shared/concat CoRA")
            if int(r) != 16:
                raise ValueError("Group projectors require CoRA rank16")
        self.rank_profile = tuple(int(v) for v in rank_profile) if rank_profile else None
        if self.rank_profile is not None:
            if self.rank_profile not in ((4, 6, 8, 10), (12, 14, 18, 20)):
                raise ValueError("Grouped CoRA rank profile must be 4-6-8-10 or 12-14-18-20")
            if injection_mode != "shared" or input_mode != "concat":
                raise ValueError("Grouped-rank CoRA requires the original shared/concat input path")
            self.r = max(self.rank_profile)
        else:
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
        if self.rank_profile is None:
            self.final = nn.Linear(user_embedding_size * 2, hidden_size * r)
            nn.init.kaiming_uniform_(self.final.weight, a=math.sqrt(5))
        elif self.rank_profile == (4, 6, 8, 10):
            # Preserve the previously evaluated compressed rank-10 architecture.
            _rank_head_rng = torch.get_rng_state()
            try:
                self.final = nn.Linear(user_embedding_size * 2, hidden_size * self.r)
                nn.init.kaiming_uniform_(self.final.weight, a=math.sqrt(5))
            finally:
                torch.set_rng_state(_rank_head_rng)
        else:
            # Parameter-matched rank profile: preserve the *exact* G2
            # 512 -> (4096*16) output head, and add only four high-rank
            # channels via a compact 512 -> 32 -> (4096*4) nonlinear branch.
            # P budget is identical to G2 and new head params are ~0.56M,
            # instead of ~8.4M for a naive dense rank-20 head.
            self.final = nn.Linear(user_embedding_size * 2, hidden_size * 16)
            nn.init.kaiming_uniform_(self.final.weight, a=math.sqrt(5))
            _extra_head_rng = torch.get_rng_state()
            try:
                self.extra_rank_head = nn.Sequential(
                    nn.Linear(user_embedding_size * 2, 32),
                    nn.GELU(),
                    nn.Linear(32, hidden_size * 4),
                )
            finally:
                torch.set_rng_state(_extra_head_rng)

        # Architecture-specific parameters must not consume the global RNG
        # stream used later by P-layer initialization and the training sampler.
        # Otherwise identical seed/config runs silently get different example
        # orders merely because one variant has extra parameters. Initialize the
        # extra modules from the current stream, then restore that stream.
        _global_cpu_rng = torch.get_rng_state()
        try:
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
        finally:
            torch.set_rng_state(_global_cpu_rng)

        # Optional NIGHT15/NIGHT16 collaborative-representation projectors.
        # All new paths act on QFormer queries before the original final head;
        # the 128 per-layer CoRA-P matrices are never replaced.
        # Every residual output starts at zero, preserving G2's initial A.
        # Restore RNG so added architecture does not alter sampler/P init.
        if self.group_projector_mode != "off":
            _projector_rng = torch.get_rng_state()
            try:
                width = self.embedding_size * 2
                if self.group_projector_mode in (
                    "shared", "group2", "group4", "group8",
                    "shared_group4", "group4_linear",
                    "coarse_to_fine", "fine_to_coarse", "hierarchical_residual",
                ):
                    counts = {
                        "shared": 1, "group2": 2, "group4": 4, "group8": 8,
                        "shared_group4": 5, "group4_linear": 4,
                        "coarse_to_fine": 8, "fine_to_coarse": 8,
                        "hierarchical_residual": 8,
                    }
                    self.group_projectors = nn.ModuleList()
                    for _ in range(counts[self.group_projector_mode]):
                        if self.group_projector_mode == "group4_linear":
                            # Parameter-matched linear bottleneck: same dimensions
                            # as the MLP variant, but no nonlinear activation.
                            projector = nn.Sequential(
                                nn.Linear(width, 32),
                                nn.Linear(32, width),
                            )
                        else:
                            projector = nn.Sequential(
                                nn.Linear(width, 32),
                                nn.GELU(),
                                nn.Linear(32, width),
                            )
                        nn.init.zeros_(projector[-1].weight)
                        nn.init.zeros_(projector[-1].bias)
                        self.group_projectors.append(projector)
                elif self.group_projector_mode == "shared_trunk4":
                    self.group_projector_trunk = nn.Linear(width, 32)
                    self.group_projector_heads = nn.ModuleList(
                        [nn.Linear(32, width) for _ in range(4)]
                    )
                    for head in self.group_projector_heads:
                        nn.init.zeros_(head.weight)
                        nn.init.zeros_(head.bias)
                elif self.group_projector_mode == "rank_group4":
                    self.group_rank_delta = nn.Parameter(
                        torch.zeros(4, self.r, self.r)
                    )
            finally:
                torch.set_rng_state(_projector_rng)

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

    def forward(self, ui_embedding, return_condition=False, collab_features=None):
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

        if hasattr(self, "n24_grad_probe"):
            self.n24_grad_probe(encoder_states, "qformer_input")
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
        # NIGHT15/NIGHT16 architecture variants. All return dynamic A matrices
        # keyed by the group id consumed by OriginalCoRARuntime.
        if self.group_projector_mode != "off":
            if return_condition:
                raise ValueError("Group-projector mode does not use return_condition")
            batch_size = encoder_states.size(0)
            q_states = query_output.last_hidden_state
            if hasattr(self, "n24_grad_probe"):
                self.n24_grad_probe(q_states, "qformer_query")
            mode = self.group_projector_mode

            def make_weight(projected):
                flat = self.final(projected).mean(dim=1, keepdim=True)
                return F.normalize(flat, p=2, dim=-1).view(
                    batch_size, self.hidden_size, self.r
                )

            if mode == "rank_group4":
                # Group-specific basis transformation after the original
                # generator head. Delta=0 is exact identity in rank space.
                raw = self.final(q_states).mean(dim=1, keepdim=True)
                raw = raw.view(batch_size, 1, self.hidden_size, self.r)
                eye = torch.eye(self.r, dtype=raw.dtype, device=raw.device)
                group_weights = {}
                for group_idx in range(4):
                    transformed = torch.matmul(
                        raw, eye + self.group_rank_delta[group_idx]
                    )
                    flat = transformed.reshape(batch_size, 1, -1)
                    group_weights[group_idx] = F.normalize(
                        flat, p=2, dim=-1
                    ).view(batch_size, self.hidden_size, self.r)
                return group_weights

            if mode == "shared_trunk4":
                shared_features = F.gelu(self.group_projector_trunk(q_states))
                group_weights = {}
                for group_idx, head in enumerate(self.group_projector_heads):
                    projected = q_states + head(shared_features)
                    group_weights[group_idx] = make_weight(projected)
                return group_weights

            if mode == "shared":
                projected = q_states + self.group_projectors[0](q_states)
                weight = make_weight(projected)
                # Keep one shared generator but expose the same A under each
                # 8-layer group id expected by the adapter/runtime.
                return {group_idx: weight for group_idx in range(4)}

            if mode == "shared_group4":
                shared_residual = self.group_projectors[0](q_states)
                group_weights = {}
                for group_idx in range(4):
                    projected = (
                        q_states + shared_residual
                        + self.group_projectors[group_idx + 1](q_states)
                    )
                    group_weights[group_idx] = make_weight(projected)
                return group_weights

            if mode == "hierarchical_residual":
                # Eight modules total: one shared residual branch and seven
                # group-specific branches. Group 0 uses only the shared branch;
                # later groups progressively increase their residual weight.
                shared_residual = self.group_projectors[0](q_states)
                group_weights = {}
                for group_idx in range(8):
                    projected = q_states + shared_residual
                    if group_idx > 0:
                        residual = self.group_projectors[group_idx](q_states)
                        projected = projected + (group_idx / 7.0) * residual
                    group_weights[group_idx] = make_weight(projected)
                return group_weights

            group_count = {
                "group2": 2, "group4": 4, "group8": 8,
                "group4_linear": 4,
                "coarse_to_fine": 8, "fine_to_coarse": 8,
            }[mode]
            group_weights = {}
            if collab_features is not None:
                if mode != "group8" or not hasattr(self, "n20_coarse_head"):
                    raise RuntimeError("NIGHT20 condition only supports original group8")
                c, f = collab_features
                # Feature widths are 2*MF_dim+2 (coarse) and 2*MF_dim+1
                # (fine). In production MF_dim=256 -> 514/513; the CPU
                # preflight intentionally uses MF_dim=16 -> 34/33.
                expected_coarse = (batch_size, 2 * self.embedding_size + 2)
                expected_fine = (batch_size, 2 * self.embedding_size + 1)
                if tuple(c.shape) != expected_coarse or tuple(f.shape) != expected_fine:
                    raise RuntimeError(
                        f"NIGHT20 feature mismatch: coarse={tuple(c.shape)} "
                        f"expected={expected_coarse}; fine={tuple(f.shape)} "
                        f"expected={expected_fine}"
                    )
                if (self.n20_coarse_head[0].in_features != expected_coarse[1]
                        or self.n20_fine_head[0].in_features != expected_fine[1]
                        or self.n20_coarse_head[-1].out_features != q_states.shape[-1]
                        or self.n20_fine_head[-1].out_features != q_states.shape[-1]):
                    raise RuntimeError(
                        "NIGHT20 condition head dimensions do not match "
                        "the MF feature width and QFormer hidden width"
                    )
                dc = self.n20_coarse_head(c).unsqueeze(1)
                df = self.n20_fine_head(f).unsqueeze(1)
                # Cap each branch to at most 10% of original QFormer q-norm.
                # Detach norm factors to avoid unnatural gradient paths
                # through the clip radius; zero-init is exactly identity.
                qnorm = q_states.detach().float().norm(dim=-1, keepdim=True).mean(dim=1, keepdim=True)
                def clip_to_q(d):
                    dn = d.detach().float().norm(dim=-1, keepdim=True)
                    factor = (self.n20_query_cap * qnorm / dn.clamp_min(1e-8)).clamp(max=1.)
                    return d * factor.to(d.dtype)
                dc, df = clip_to_q(dc), clip_to_q(df)
            for group_idx in range(group_count):
                if collab_features is not None:
                    alpha, beta = self.n20_schedule[group_idx]
                    # Insert coarse/fine conditions BEFORE the unchanged
                    # group8 projector and final dynamic-A head.
                    projected_in = q_states + alpha * dc + beta * df
                else:
                    projected_in = q_states
                projected = projected_in + self.group_projectors[group_idx](projected_in)
                group_weights[group_idx] = make_weight(projected)
                if hasattr(self, "n24_grad_probe") and group_idx in (0, 7):
                    self.n24_grad_probe(
                        group_weights[group_idx], f"cora_group_{group_idx}"
                    )
            return group_weights

        # All four layers groups reuse one QFormer representation and a
        # shared collection of rank channels; each group takes a prefix and
        # normalizes the selected channels, as in original CoRA.
        if self.rank_profile is not None:
            if return_condition:
                raise ValueError("return_condition is not used by grouped-rank CoRA")
            batch_size = encoder_states.size(0)
            q_states = query_output.last_hidden_state
            raw = self.final(q_states).mean(dim=1, keepdim=True)
            if self.rank_profile == (12, 14, 18, 20):
                # G2-compatible first 16 channels plus four extra channels;
                # extra head is nonlinear, so it is not merely a linear
                # re-expression of the existing 16 channels.
                base = raw.view(batch_size, 1, self.hidden_size, 16)
                extras = self.extra_rank_head(q_states).mean(
                    dim=1, keepdim=True
                ).view(batch_size, 1, self.hidden_size, 4)
                raw = torch.cat((base, extras), dim=-1)
            else:
                raw = raw.view(batch_size, 1, self.hidden_size, self.r)
            grouped_weights = {}
            for rank in self.rank_profile:
                flat = raw[..., :rank].reshape(batch_size, 1, self.hidden_size * rank)
                grouped_weights[rank] = F.normalize(flat, p=2, dim=-1).view(
                    batch_size, self.hidden_size, rank
                )
            return grouped_weights

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
