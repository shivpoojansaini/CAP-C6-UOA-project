import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0


class SpatialIdentityAttnProcessor(AttnProcessor2_0):
    def __init__(self, identity_token_indices, base_masks):
        """
        identity_token_indices: List[List[int]]
            Token indices for each identity (e.g. [[6,7],[9,10]])

        base_masks: Tensor [num_id, H, W]
            Binary masks in full latent resolution
        """
        super().__init__()


        self.identity_token_indices = identity_token_indices
        self.base_masks = base_masks  # [N, H, W]
        # 🔥 add these
        self.current_step = 0
        self.total_steps = 50
        self.start_merge_step = 10
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

        # --------------------------------------------------
        # 1️⃣ Self-attention fallback
        # --------------------------------------------------
        if encoder_hidden_states is None:
            return super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
            )

        # --------------------------------------------------
        # 2️⃣ Standard QKV
        # --------------------------------------------------
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

        # reshape → [B, heads, spatial, text]
        attn_scores = attn_scores.view(
            batch_size, heads, spatial_tokens, text_tokens
        )
        # --------------------------------------------------
        # 🔥 Identity Logit Bias (CFG-Aware, Controlled)
        # --------------------------------------------------

        identity_bias = 1.0  # Start small: 0.8 → 1.5 range

        if batch_size % 2 == 0:
            half = batch_size // 2
            target_slice = slice(half, batch_size)  # conditional branch
        else:
            target_slice = slice(0, batch_size)

        for token_indices in self.identity_token_indices:
            if len(token_indices) == 0:
                continue

            attn_scores[target_slice, :, :, token_indices] += identity_bias
        # --------------------------------------------------
        # 3️⃣ Downsample masks to current spatial size
        # --------------------------------------------------
        spatial_size = int(spatial_tokens ** 0.5)

        masks = F.interpolate(
            self.base_masks.unsqueeze(1),  # [N,1,H,W]
            size=(spatial_size, spatial_size),
            mode="nearest",
        ).squeeze(1)  # [N, h, w]

        masks = masks.reshape(masks.shape[0], -1)  # [N, spatial_tokens]
        masks = (masks > 0).to(attn_scores.dtype)  # binary mask
        masks = masks.to(attn_scores.device)

        # --------------------------------------------------
        # 4️⃣ Global Softmax FIRST
        # --------------------------------------------------
        attn_scores = attn_scores.view(B_heads, spatial_tokens, text_tokens)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = attn_probs.view(batch_size, heads, spatial_tokens, text_tokens)

        # --------------------------------------------------
        # CFG Split (only modify conditional half)
        # --------------------------------------------------

        if batch_size % 2 == 0:
            half = batch_size // 2
            target_slice = slice(half, batch_size)  # conditional branch
        else:
            target_slice = slice(0, batch_size)  # no CFG


        # --------------------------------------------------
        # 5️⃣ Strong Probability-Space Routing (Stable Scheduled)
        # --------------------------------------------------

        if self.current_step < self.start_merge_step:
            IDENTITY_REGION_MASS = 0.0
        else:
            # Safe denominator
            denom = max(self.total_steps - self.start_merge_step, 1)

            progress = (self.current_step - self.start_merge_step) / denom
            progress = float(min(max(progress, 0.0), 1.0))

            # 🔥 Stable 3-phase schedule
            if progress < 0.20:
                IDENTITY_REGION_MASS = 0.72
            elif progress < 0.75:
                IDENTITY_REGION_MASS = 0.85
            else:
                IDENTITY_REGION_MASS = 0.80
            # Relax slightly for texture stabilization




        for identity_i, token_indices in enumerate(self.identity_token_indices):

            if len(token_indices) == 0:
                continue

            identity_mask = masks[identity_i]
            identity_mask = identity_mask.view(1, 1, spatial_tokens, 1)

            # Slice identity + non identity
            identity_probs = attn_probs[target_slice, :, :, token_indices]

            non_identity_mask = torch.ones(text_tokens, device=attn_probs.device)
            non_identity_mask[token_indices] = 0
            non_identity_mask = non_identity_mask.view(1,1,1,text_tokens)

            non_identity_probs = attn_probs[target_slice] * non_identity_mask

            # Sum inside region
            id_sum = identity_probs.sum(dim=-1, keepdim=True) + 1e-8
            non_id_sum = non_identity_probs.sum(dim=-1, keepdim=True) + 1e-8

            # Normalize each group
            identity_probs = identity_probs / id_sum
            non_identity_probs = non_identity_probs / non_id_sum

            # Reassign mass
            identity_probs = identity_probs * IDENTITY_REGION_MASS
            non_identity_probs = non_identity_probs * (1 - IDENTITY_REGION_MASS)

            # Combine
            combined = torch.zeros_like(attn_probs[target_slice])
            combined[..., token_indices] = identity_probs
            combined += non_identity_probs

            # Apply only inside region
            attn_probs[target_slice] = (
                combined * identity_mask
                + attn_probs[target_slice] * (1 - identity_mask)
            )



        # --------------------------------------------------
        # 6️⃣ Renormalize (CRITICAL)
        # --------------------------------------------------
        attn_probs_sum = attn_probs.sum(dim=-1, keepdim=True) + 1e-8
        attn_probs = attn_probs / attn_probs_sum

        # --------------------------------------------------
        # 7️⃣ Final Projection
        # --------------------------------------------------
        attn_probs = attn_probs.view(B_heads, spatial_tokens, text_tokens)

        hidden_states = torch.bmm(attn_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
