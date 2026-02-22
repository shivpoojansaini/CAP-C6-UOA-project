import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0


class SpatialBiasAttnProcessor(AttnProcessor2_0):
    def __init__(
        self,
        identity_token_indices,
        base_masks,
        identity_bias=2.0,
        spatial_strength=0.8,
        outside_suppress=0.6,
    ):
        super().__init__()

        self.identity_token_indices = identity_token_indices
        self.base_masks = base_masks  # [N, H, W]

        self.identity_bias = identity_bias
        self.spatial_strength = spatial_strength
        self.outside_suppress = outside_suppress

        # keep for pipeline compatibility
        self.current_step = 0
        self.total_steps = 50

    def set_step(self, step, total_steps):
        self.current_step = step
        self.total_steps = total_steps

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):

        # ---------------------------------------
        # 1️⃣ Self-attention fallback
        # ---------------------------------------
        if encoder_hidden_states is None:
            return super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
            )

        # ---------------------------------------
        # 2️⃣ Standard QKV
        # ---------------------------------------
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_scores = torch.bmm(query, key.transpose(-1, -2))
        attn_scores *= attn.scale

        B_heads, spatial_tokens, text_tokens = attn_scores.shape
        heads = attn.heads
        batch_size = B_heads // heads

        attn_scores = attn_scores.view(
            batch_size, heads, spatial_tokens, text_tokens
        )

        # ---------------------------------------
        # 3️⃣ CFG-safe conditional slice
        # ---------------------------------------
        if batch_size % 2 == 0:
            half = batch_size // 2
            target_slice = slice(half, batch_size)
        else:
            target_slice = slice(0, batch_size)

        # ---------------------------------------
        # 4️⃣ Downsample masks to current spatial resolution
        # ---------------------------------------
        spatial_size = int(spatial_tokens ** 0.5)

        masks = F.interpolate(
            self.base_masks.unsqueeze(1),  # [N,1,H,W]
            size=(spatial_size, spatial_size),
            mode="nearest",
        ).squeeze(1)

        masks = masks.reshape(masks.shape[0], -1)  # [N, spatial_tokens]
        masks = masks.to(attn_scores.device)

        # ---------------------------------------
        # 5️⃣ Spatial Logit Bias (Balanced Version)
        # ---------------------------------------
        base_bias = self.identity_bias * self.spatial_strength

        for identity_i, token_indices in enumerate(self.identity_token_indices):

            if len(token_indices) == 0:
                continue

            identity_mask = masks[identity_i]  # [spatial_tokens]
            identity_mask = identity_mask.view(1, 1, spatial_tokens, 1)

            # ---------------------------------------
            # 🔥 Per-Identity Compensation
            # ---------------------------------------

            identity_specific_bias = base_bias

        

            # ---------------------------------------
            # Inside-region boost
            # ---------------------------------------

            attn_scores[target_slice, :, :, token_indices] += (
                identity_specific_bias * identity_mask
            )

            # ---------------------------------------
            # Outside-region mild suppression
            # ---------------------------------------

            attn_scores[target_slice, :, :, token_indices] -= (
                self.outside_suppress
                * identity_specific_bias
                * (1 - identity_mask)
            )


        # ---------------------------------------
        # 6️⃣ Softmax + projection
        # ---------------------------------------
        attn_scores = attn_scores.view(B_heads, spatial_tokens, text_tokens)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        hidden_states = torch.bmm(attn_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
