import torch
from diffusers.models.attention_processor import AttnProcessor2_0


class IdentityBiasAttnProcessor(AttnProcessor2_0):
    """
    Simple identity logit bias processor.
    No spatial masking.
    No probability mass reassignment.
    Just adds bias to identity tokens in cross-attention.
    """

    def __init__(self, identity_token_indices, identity_bias=2.0):
        super().__init__()
        self.identity_token_indices = identity_token_indices
        self.identity_bias = identity_bias

    def set_step(self, step, total_steps):
        # Not used for bias-only processor
        pass
    
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
        # 3️⃣ Apply Identity Logit Bias (Conditional Half Only)
        # --------------------------------------------------
        if batch_size % 2 == 0:
            half = batch_size // 2
            target_slice = slice(half, batch_size)  # conditional branch
        else:
            target_slice = slice(0, batch_size)  # no CFG

        for token_indices in self.identity_token_indices:
            if len(token_indices) == 0:
                continue

            attn_scores[target_slice, :, :, token_indices] += self.identity_bias

        # --------------------------------------------------
        # 4️⃣ Softmax
        # --------------------------------------------------
        attn_scores = attn_scores.view(B_heads, spatial_tokens, text_tokens)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # --------------------------------------------------
        # 5️⃣ Projection
        # --------------------------------------------------
        hidden_states = torch.bmm(attn_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
