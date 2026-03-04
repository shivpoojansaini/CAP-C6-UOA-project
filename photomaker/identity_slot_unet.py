import torch
import torch.nn as nn


class IdentitySlotUNet(nn.Module):
    """
    Competitive Multi-Identity Injection UNet Wrapper

    Supports:
    • Stream-aware slot activation
    • Optional injection disable (analysis mode)
    • Identity-feature similarity logging
    • UNet feature probing
    """

    def __init__(
        self,
        unet,
        down_strength=0.0,
        mid_strength=0.0,
        up_strength=0.0,
        temperature=1.0,
    ):
        super().__init__()

        self.unet = unet
        self.down_strength = down_strength
        self.mid_strength = mid_strength
        self.up_strength = up_strength
        self.temperature = temperature

        # Debug controls
        self.disable_injection = True
        self.debug_similarity = []
        self.current_step = 0
        self.last_hidden_states = None

        # Identity state
        self.identity_embeddings = None
        self.identity_bboxes = None
        self.active_slots = None

        # 🔥 IMPORTANT: projection starts as None
        self.proj = None

        self.config = unet.config

        self._register_hooks()

    # -------------------------------------------------
    # Required properties for diffusers
    # -------------------------------------------------

    @property
    def device(self):
        return next(self.unet.parameters()).device

    @property
    def dtype(self):
        return next(self.unet.parameters()).dtype

    # -------------------------------------------------
    # Identity setters
    # -------------------------------------------------

    def set_identity_data(self, embeddings, bboxes):
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        self.identity_embeddings = embeddings
        self.identity_bboxes = bboxes

    def set_active_slots(self, active_indices):
        self.active_slots = active_indices

    def clear_identity_data(self):
        self.identity_embeddings = None
        self.identity_bboxes = None
        self.active_slots = None
        self.debug_similarity = []

    # -------------------------------------------------
    # Core Injection Logic
    # -------------------------------------------------

    def _inject(self, hidden_states, strength, log_similarity=False):

        # -------------------------------------------------
        # Early exits
        # -------------------------------------------------

        if (
            self.identity_embeddings is None
            or self.identity_bboxes is None
            or not isinstance(hidden_states, torch.Tensor)
        ):
            return hidden_states

        if hidden_states.dim() != 4:
            return hidden_states

        B, C, H, W = hidden_states.shape

        # 🔬 Store hidden states for probing
        if log_similarity:
            self.last_hidden_states = hidden_states.detach()

        # -------------------------------------------------
        # Normalize hidden features
        # -------------------------------------------------

        hidden_flat = hidden_states.flatten(2)  # [B, C, HW]
        hidden_flat = hidden_flat / (hidden_flat.norm(dim=1, keepdim=True) + 1e-6)

        B_id, N_total, D = self.identity_embeddings.shape
        if N_total == 0:
            return hidden_states

        # CFG-safe repeat
        identity_embeddings = (
            self.identity_embeddings.repeat(B, 1, 1)
            if B != B_id
            else self.identity_embeddings
        )

        # Active slots
        active_indices = (
            self.active_slots
            if self.active_slots and len(self.active_slots) > 0
            else list(range(N_total))
        )

        identity_embeddings = identity_embeddings[:, active_indices]
        identity_bboxes = self.identity_bboxes[active_indices]

        B, N, D = identity_embeddings.shape

        # -------------------------------------------------
        # 🔥 Lazy projection initialization (CRITICAL FIX)
        # -------------------------------------------------

        if self.proj is None:
            self.proj = nn.Sequential(
                nn.Linear(512, C),
                nn.GELU(),
                nn.Linear(C, C),
            ).to(hidden_states.device, hidden_states.dtype)

        projected = self.proj(identity_embeddings.reshape(B * N, D))
        projected = projected.view(B, N, C)

        projected = projected / (projected.norm(dim=2, keepdim=True) + 1e-6)
        projected = projected * (C ** 0.5)

        # -------------------------------------------------
        # Spatial masks
        # -------------------------------------------------

        masks = torch.zeros(
            (B, N, 1, H, W),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for i in range(N):
            x1, y1, x2, y2 = identity_bboxes[i]

            x1_i = int(max(0, min(W, x1.item() * W)))
            x2_i = int(max(0, min(W, x2.item() * W)))
            y1_i = int(max(0, min(H, y1.item() * H)))
            y2_i = int(max(0, min(H, y2.item() * H)))

            if x2_i > x1_i and y2_i > y1_i:
                masks[:, i, :, y1_i:y2_i, x1_i:x2_i] = 1.0

        if masks.sum() == 0:
            return hidden_states

        # -------------------------------------------------
        # Cosine similarity
        # -------------------------------------------------

        scores = torch.einsum("bnc,bch->bnh", projected, hidden_flat)
        scores = scores.view(B, N, H, W)
        scores = scores * masks.squeeze(2)

        if log_similarity:
            self.debug_similarity.append({
                "step": getattr(self, "current_step", -1),
                "scores": scores.detach().cpu()
            })

        # -------------------------------------------------
        # Analysis mode
        # -------------------------------------------------

        if self.disable_injection:
            return hidden_states

        # -------------------------------------------------
        # Competitive weighting
        # -------------------------------------------------

        weights = torch.sigmoid(scores / (self.temperature + 1e-6))
        weights = weights * masks.squeeze(2)

        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-6
        weights = weights / weights_sum

        projected = projected.view(B, N, C, 1, 1)
        weights = weights.unsqueeze(2)

        total_injection = (weights * projected).sum(dim=1)

        hidden_states = hidden_states + strength * total_injection

        return hidden_states

    # -------------------------------------------------
    # Hooks
    # -------------------------------------------------

    def _up_block_hook(self, module, input, output):

        if isinstance(output, tuple):
            hidden = output[0]
            injected = self._inject(
                hidden,
                self.up_strength,
                log_similarity=True,
            )
            return (injected,) + output[1:]
        else:
            return self._inject(
                output,
                self.up_strength,
                log_similarity=True,
            )

    def _register_hooks(self):
        # Hook ONLY last up block (highest resolution)
        self.unet.up_blocks[-1].register_forward_hook(
            self._up_block_hook
        )

    # -------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)

    # -------------------------------------------------

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)