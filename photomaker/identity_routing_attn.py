import torch
from diffusers.models.attention_processor import AttnProcessor2_0


class IdentityPromptRoutingProcessor(AttnProcessor2_0):
    """
    Identity → Prompt Token Routing Processor

    Forces each identity token group to attend only to
    specific prompt word tokens.

    Args:
        identity_token_indices: List[List[int]]
            Example: [[4,5], [9,10]]

        identity_prompt_map: Dict[int, List[int]]
            Example:
            {
                0: [12, 13],      # "man sunglasses"
                1: [15, 16],      # "woman spacesuit"
            }

        routing_strength: float
            How strongly to suppress disallowed tokens.
            Recommended: 6.0 → 12.0
    """

    def __init__(
        self,
        identity_token_indices,
        identity_prompt_map,
        routing_strength=8.0,
    ):
        super().__init__()

        self.identity_token_indices = identity_token_indices
        self.identity_prompt_map = identity_prompt_map
        self.routing_strength = routing_strength

        # Needed so pipeline does not crash if set_step is called
        self.current_step = 0
        self.total_steps = 50

    # ----------------------------------------------------
    # Diffusers step hook
    # ----------------------------------------------------
    def set_step(self, step, total_steps):
        self.current_step = step
        self.total_steps = total_steps

    # ----------------------------------------------------
    # Forward
    # ----------------------------------------------------
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):

        # -----------------------------------------
        # Self-attention fallback
        # -----------------------------------------
        if encoder_hidden_states is None:
            return super().__call__(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                temb,
            )

        # -----------------------------------------
        # Standard Cross Attention Computation
        # -----------------------------------------
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
            batch_size,
            heads,
            spatial_tokens,
            text_tokens,
        )

        # -----------------------------------------
        # CFG-safe slice (modify only conditional half)
        # -----------------------------------------
        if batch_size % 2 == 0:
            half = batch_size // 2
            target_slice = slice(half, batch_size)
        else:
            target_slice = slice(0, batch_size)

        # -----------------------------------------
        # 🔥 Identity → Prompt Routing
        # -----------------------------------------

        large_neg = -self.routing_strength

        for identity_i, id_token_indices in enumerate(self.identity_token_indices):

            if len(id_token_indices) == 0:
                continue

            allowed_prompt_tokens = self.identity_prompt_map.get(identity_i, [])

            if len(allowed_prompt_tokens) == 0:
                continue

            # Create mask of disallowed prompt tokens
            disallowed_mask = torch.ones(
                text_tokens,
                device=attn_scores.device,
                dtype=attn_scores.dtype,
            )

            disallowed_mask[allowed_prompt_tokens] = 0.0

            # Shape: [1, 1, 1, text_tokens]
            disallowed_mask = disallowed_mask.view(1, 1, 1, text_tokens)

            # Apply routing only to identity token channels
            # We modify attention scores for all spatial positions
            # but only for the conditional batch slice
            attn_scores[target_slice] += disallowed_mask * large_neg

        # -----------------------------------------
        # Softmax
        # -----------------------------------------
        attn_scores = attn_scores.view(B_heads, spatial_tokens, text_tokens)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        hidden_states = torch.bmm(attn_probs, value)

        # Restore shape
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
