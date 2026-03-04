import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0


class SpatialRoutingProcessor(AttnProcessor2_0):

    def __init__(
        self,
        identity_token_indices,
        identity_prompt_map,
        base_masks,
        tokenizer=None,
        text_input_ids=None,
        identity_bias=2.5,
        spatial_strength=1.0,
        outside_suppress=1.5,
        routing_strength=6.0,
        cross_identity_strength=8.0,
    ):
        super().__init__()

        self.identity_token_indices = identity_token_indices
        self.identity_prompt_map = identity_prompt_map
        self.base_masks = base_masks

        self.tokenizer = tokenizer
        self.processor_text_input_ids = text_input_ids

        self.identity_bias = identity_bias
        self.spatial_strength = spatial_strength
        self.outside_suppress = outside_suppress
        self.routing_strength = routing_strength
        self.cross_identity_strength = cross_identity_strength

        self.current_step = 0
        self.total_steps = 50

    # ---------------------------------------------------------
    # Diffusion step setter
    # ---------------------------------------------------------
    def set_step(self, step, total_steps):
        self.current_step = step
        self.total_steps = total_steps

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):

        # Self-attention untouched
        if encoder_hidden_states is None:
            return super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
            )

        # QKV
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Attention logits
        attn_scores = torch.bmm(query, key.transpose(-1, -2))
        attn_scores *= attn.scale

        B_heads, spatial_tokens, text_tokens = attn_scores.shape
        heads = attn.heads
        batch_size = B_heads // heads

        attn_scores = attn_scores.view(
            batch_size, heads, spatial_tokens, text_tokens
        )

        # CFG-safe slice
        if batch_size % 2 == 0:
            half = batch_size // 2
            target_slice = slice(half, batch_size)
        else:
            target_slice = slice(0, batch_size)

        # Downsample masks
        spatial_size = int(spatial_tokens ** 0.5)

        masks = F.interpolate(
            self.base_masks.unsqueeze(1),
            size=(spatial_size, spatial_size),
            mode="nearest",
        ).squeeze(1)

        masks = masks.reshape(masks.shape[0], -1)
        masks = masks.float().to(attn_scores.device)

        # Early-heavy schedule
        progress = 1 - (self.current_step / max(self.total_steps, 1))

        identity_boost = self.identity_bias * progress
        routing_boost = self.routing_strength * progress
        cross_penalty = self.cross_identity_strength * progress
        outside_penalty = self.outside_suppress

        # ---------------------------------------------------------
        # Spatial + Prompt Routing
        # ---------------------------------------------------------
        for identity_i, token_indices in enumerate(self.identity_token_indices):

            if len(token_indices) == 0:
                continue

            identity_mask = masks[identity_i]
            identity_mask = identity_mask.view(1, 1, spatial_tokens, 1)

            # Boost own identity tokens inside region
            attn_scores[target_slice, :, :, token_indices] += (
                identity_boost * identity_mask
            )

            # Suppress identity tokens outside region
            attn_scores[target_slice, :, :, token_indices] -= (
                outside_penalty * (1 - identity_mask)
            )

            # Boost own attribute span
            allowed_tokens = self.identity_prompt_map.get(identity_i, [])
            if len(allowed_tokens) > 0:
                attn_scores[target_slice, :, :, allowed_tokens] += (
                    routing_boost * identity_mask
                )

            # Suppress other identities softly (not hard mask)
            for other_i, other_tokens in enumerate(self.identity_token_indices):
                if other_i == identity_i:
                    continue

                attn_scores[target_slice, :, :, other_tokens] -= (
                    cross_penalty * identity_mask
                )

                other_span = self.identity_prompt_map.get(other_i, [])
                if len(other_span) > 0:
                    attn_scores[target_slice, :, :, other_span] -= (
                        cross_penalty * identity_mask
                    )

        # Softmax
        attn_scores = attn_scores.view(B_heads, spatial_tokens, text_tokens)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        hidden_states = torch.bmm(attn_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states