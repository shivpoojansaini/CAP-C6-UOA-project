import torch
import torch.nn as nn


class IdentitySlotUNet(nn.Module):
    """
    Competitive Multi-Identity Injection UNet Wrapper
    Supports:
    • Stream-aware slot activation
    • Optional injection disable (analysis mode)
    • Identity-feature similarity logging
    """

    def __init__(
        self,
        unet,
        down_strength=0.0,
        mid_strength=1.5,
        up_strength=1.5,
        temperature=0.5,
    ):
        super().__init__()

        self.unet = unet
        self.down_strength = down_strength
        self.mid_strength = mid_strength
        self.up_strength = up_strength
        self.temperature = temperature

        # 🔬 Debug controls
        self.disable_injection = True
        self.debug_similarity = []   # stores similarity maps
        self.debug_step = 0

        # Identity state
        self.identity_embeddings = None       # [B, N, 512]
        self.identity_bboxes = None           # [N, 4]
        self.active_slots = None

        # Projection to UNet feature dimension
        latent_channels = unet.config.block_out_channels[-1]

        self.proj = nn.Sequential(
            nn.Linear(512, latent_channels),
            nn.GELU(),
            nn.Linear(latent_channels, latent_channels),
        )

        # Match UNet precision + device
        self.proj.to(unet.device, dtype=unet.dtype)

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
        """
        embeddings: [B, N, 512]
        bboxes:     [N, 4] normalized (x1,y1,x2,y2)
        """
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
    # Hook registration
    # -------------------------------------------------



        

    # -------------------------------------------------
    # Core Injection Logic
    # -------------------------------------------------

    def _inject(self, hidden_states, strength, log_similarity=False):
        """
        hidden_states: [B, C, H, W] OR [B, C, HW]
        """

        # -------------------------------------------------
        # Early safety checks
        # -------------------------------------------------

        if (
            self.identity_embeddings is None
            or self.identity_bboxes is None
            or not isinstance(hidden_states, torch.Tensor)
        ):
            return hidden_states

        # -------------------------------------------------
        # Safe hidden state flattening
        # -------------------------------------------------

        if hidden_states.dim() == 4:
            B, C, H, W = hidden_states.shape
            hidden_flat = hidden_states.flatten(2)  # [B, C, H*W]
            is_spatial = True
        elif hidden_states.dim() == 3:
            B, C, HW = hidden_states.shape
            hidden_flat = hidden_states
            is_spatial = False
        else:
            return hidden_states
        C_hidden = hidden_flat.shape[1]
        # Normalize hidden features
        hidden_flat = hidden_flat / (
            hidden_flat.norm(dim=1, keepdim=True) + 1e-6
        )

        # -------------------------------------------------
        # Handle identity embeddings (CFG-safe)
        # -------------------------------------------------

        B_id, N_total, D = self.identity_embeddings.shape
        if N_total == 0:
            return hidden_states

        if B != B_id:
            identity_embeddings = self.identity_embeddings.repeat(B, 1, 1)
        else:
            identity_embeddings = self.identity_embeddings

        # -------------------------------------------------
        # Determine active slots
        # -------------------------------------------------

        if self.active_slots and len(self.active_slots) > 0:
            active_indices = self.active_slots
        else:
            active_indices = list(range(N_total))

        identity_embeddings = identity_embeddings[:, active_indices]
        identity_bboxes = self.identity_bboxes[active_indices]

        B, N, D = identity_embeddings.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        # ---------------------------------------------
        # 🔬 Debug: identity embedding similarity
        # ---------------------------------------------
        if log_similarity and N >= 2:
            cos_sim = torch.cosine_similarity(
                identity_embeddings[:, 0],
                identity_embeddings[:, 1],
                dim=-1
            ).mean().item()

            print(f"🔍 Cosine similarity between ID 0 and ID 1: {cos_sim:.4f}")
        # -------------------------------------------------
        # 1️⃣ Project identity embeddings to match hidden C
        # -------------------------------------------------

        proj_out = self.proj(identity_embeddings.reshape(B * N, D))

        # If projection dimension != hidden feature dimension
        if proj_out.shape[-1] != C_hidden:
            proj_out = nn.Linear(proj_out.shape[-1], C_hidden).to(
                proj_out.device, proj_out.dtype
            )(proj_out)

        projected = proj_out.view(B, N, C_hidden)

        projected = projected / (
            projected.norm(dim=2, keepdim=True) + 1e-6
        )
        projected = projected * (C_hidden ** 0.5)

        # -------------------------------------------------
        # 2️⃣ Build spatial masks (if spatial)
        # -------------------------------------------------

        if is_spatial:
            masks = torch.zeros((B, N, 1, H, W), device=device, dtype=dtype)

            for i in range(N):
                x1, y1, x2, y2 = identity_bboxes[i]

                x1_i = int(max(0, min(W, x1.item() * W)))
                x2_i = int(max(0, min(W, x2.item() * W)))
                y1_i = int(max(0, min(H, y1.item() * H)))
                y2_i = int(max(0, min(H, y2.item() * H)))

                if x2_i > x1_i and y2_i > y1_i:
                    masks[:, i, :, y1_i:y2_i, x1_i:x2_i] = 1.0

            if masks.sum() == 0:
                masks = torch.ones((B, N, 1, H, W), device=device, dtype=dtype)
        else:
            masks = None

        # -------------------------------------------------
        # 3️⃣ Cosine similarity
        # -------------------------------------------------

        scores = torch.einsum("bnc,bch->bnh", projected, hidden_flat)

        if is_spatial:
            scores = scores.view(B, N, H, W)
            scores = scores * masks.squeeze(2)

        # -------------------------------------------------
        # 🔬 Log FULL score tensor for argmax experiment
        # -------------------------------------------------

        if log_similarity and is_spatial:
            self.debug_similarity.append({
                "step": getattr(self, "current_step", -1),
                "scores": scores.detach().cpu()   # [B,N,H,W]
            })

        # -------------------------------------------------
        # 🚫 Analysis Mode (no injection)
        # -------------------------------------------------

        if self.disable_injection:
            return hidden_states

        # -------------------------------------------------
        # 4️⃣ Competitive weighting (soft injection)
        # -------------------------------------------------

        temperature = getattr(self, "temperature", 1.0)

        if is_spatial:
            weights = torch.sigmoid(scores / (temperature + 1e-6))
            weights = weights * masks.squeeze(2)

            weights_sum = weights.sum(dim=1, keepdim=True) + 1e-6
            weights = weights / weights_sum
            weights = weights ** 1.2

            # -------------------------------------------------
            # 5️⃣ Add injection
            # -------------------------------------------------

            projected = projected.view(B, N, C_proj, 1, 1)
            weights_expanded = weights.unsqueeze(2)

            total_injection = (weights_expanded * projected).sum(dim=1)

            hidden_states = hidden_states + strength * total_injection

        return hidden_states

    # -------------------------------------------------
    # Hooks
    # -------------------------------------------------

    def _mid_block_hook(self, module, input, output):

        if isinstance(output, tuple):
            hidden = output[0]

            injected = self._inject(
                hidden,
                self.mid_strength,
                log_similarity=False,   # 🔴 IMPORTANT: no logging here
            )

            return (injected,) + output[1:]

        else:
            return self._inject(
                output,
                self.mid_strength,
                log_similarity=False,   # 🔴 IMPORTANT
            )

    def _up_block_hook(self, module, input, output):

        if isinstance(output, tuple):
            hidden = output[0]

            injected = self._inject(
                hidden,
                self.up_strength,
                log_similarity=True,   # ✅ LOG HERE
            )

            return (injected,) + output[1:]

        else:
            return self._inject(
                output,
                self.up_strength,
                log_similarity=True,   # ✅ LOG HERE
            )


    def _register_hooks(self):
        # Remove mid-block hook if previously registered
        # (optional but recommended during experiment)

        # Register hook on last up block only
        self.unet.up_blocks[-1].register_forward_hook(
            self._up_block_hook
        )

    # -------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)

    # -------------------------------------------------
    # Delegate missing attributes
    # -------------------------------------------------

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)