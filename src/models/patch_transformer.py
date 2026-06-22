"""Patch transformer backbone.

Turns an image into a sequence of per-patch hidden states:

    image (B, C, H, W)
      -> patchify           -> (B, N, C*p*p)
      -> linear patch embed -> tokens (B, N, D)
      -> + positional embed
      -> N standard transformer blocks (multi-head self-attention + MLP, pre-norm)
      -> final norm
      -> per-patch hidden states (B, N, D)

Patches attend to each other through self-attention, so each output token mixes information
from the whole image. This is the Phase B backbone; a generative head (diffusion / flow) is
conditioned on these hidden states later.

Two modern attention tweaks are wired as *hooks* but intentionally NOT implemented yet:
  - RoPE (rotary position embedding) — `pos_encoding="rope"`.
  - QK-Norm (normalize Q/K before the attention scores) — `qk_norm=True`.
Both raise NotImplementedError if requested; the insertion points are marked in Attention so they
can be filled in without touching the forward backbone.

On top of the backbone, `PatchReconstructionModel` adds a per-patch *conditional generative head*
that reconstructs each clean patch from noise, conditioned on that patch's hidden state:

    image -> backbone -> per-patch hidden states (the bottleneck / "code")
          -> patchify(image) gives the clean target patches
          -> head reconstructs each patch from noise, conditioned on its hidden state.

This is a true encode->decode reconstruction (the hidden state is the bottleneck under study), so
the reconstruction is paired with the original image and PSNR/SSIM/LPIPS are directly meaningful.
The head is swappable behind the `PatchGenerativeHead` interface: `DiffusionPatchHead` (reusing
diffusion.py's schedule / q_sample / reverse chain) and `FlowMatchingPatchHead` (straight-line
interpolant + Euler sampling) share the same `loss` / `sample` signature, so `PatchReconstructionModel`
toggles between them with a single `head_kind` argument for a fair diffusion-vs-flow comparison.

Tensor layout matches src/data/patchify.py:
    image : (B, C, H, W)
    patch : (B, N, C*p*p),  N = (H // p) * (W // p), row-major.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.data.patchify import patchify, unpatchify
from src.models.diffusion import Diffusion, SinusoidalTimeEmbedding


class PatchEmbed(nn.Module):
    """Linearly project each flattened patch (C*p*p) into a D-dim token."""

    def __init__(self, patch_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(patch_dim, dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:  # (B, N, C*p*p) -> (B, N, D)
        return self.proj(patches)


class Attention(nn.Module):
    """Standard multi-head self-attention; every patch attends to every patch (bidirectional)."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = False, pos_encoding: str = "learned"):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # --- HOOK: QK-Norm. When enabled these become RMSNorm(head_dim); identity keeps it a no-op. ---
        self.qk_norm = qk_norm
        self.q_norm: nn.Module = nn.Identity()
        self.k_norm: nn.Module = nn.Identity()
        if qk_norm:
            raise NotImplementedError("QK-Norm hook reserved; not implemented yet")

        # --- HOOK: RoPE. Rotary embedding applied to Q/K below; reserved for now. ---
        self.pos_encoding = pos_encoding
        if pos_encoding == "rope":
            raise NotImplementedError("RoPE hook reserved; not implemented yet")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, D) -> (B, N, D)
        b, n, d = x.shape
        # (B, N, 3D) -> 3 x (B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # --- hook point: QK-Norm acts here (after head split, before scores) ---
        q, k = self.q_norm(q), self.k_norm(k)
        # --- hook point: RoPE would rotate q, k here (after QK-Norm, before SDPA) ---

        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim), no mask
        out = out.transpose(1, 2).reshape(b, n, d)     # back to (B, N, D)
        return self.proj(out)


class MLP(nn.Module):
    """Standard transformer feed-forward: D -> mlp_ratio*D -> D with GELU."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + Attn(norm(x)); x + MLP(norm(x))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qk_norm: bool, pos_encoding: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qk_norm=qk_norm, pos_encoding=pos_encoding)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchTransformer(nn.Module):
    """Image -> per-patch hidden states (B, N, D).

    Args:
        img_size:     input side length (H == W); used only to size the positional embedding.
        patch_size:   side length p of each square patch; img_size must be divisible by p.
        channels:     image channel count.
        dim:          token (hidden) dimension D.
        depth:        number of transformer blocks.
        num_heads:    attention heads (must divide dim).
        mlp_ratio:    feed-forward expansion factor.
        pos_encoding: "learned" (implemented) or "rope" (hook, not implemented).
        qk_norm:      QK-Norm hook (not implemented; True raises).
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        channels: int = 3,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        pos_encoding: str = "learned",
        qk_norm: bool = False,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.channels = channels
        self.dim = dim

        # Token count derived from the configured image/patch size — not hardcoded.
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_embed = PatchEmbed(patch_dim, dim)

        # Learned positional embedding, length = current token count.
        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
            nn.init.normal_(self.pos_embed, std=0.02)
        elif pos_encoding == "rope":
            self.pos_embed = None  # handled inside attention (hook, not implemented)
        else:
            raise ValueError(f"unknown pos_encoding '{pos_encoding}'")

        self.blocks = nn.ModuleList(
            TransformerBlock(dim, num_heads, mlp_ratio, qk_norm, pos_encoding) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")
        if h != self.img_size or w != self.img_size:
            raise ValueError(f"expected {self.img_size}x{self.img_size} images, got {h}x{w}")

        patches = patchify(images, self.patch_size)  # (B, N, C*p*p)
        x = self.patch_embed(patches)                 # (B, N, D)
        if self.pos_embed is not None:
            x = x + self.pos_embed                    # length already matches N
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)                     # (B, N, D) per-patch hidden states


# --------------------------------------------------------------------------- #
# Per-patch conditional generative head (swappable: diffusion now, flow later)
# --------------------------------------------------------------------------- #
class PatchGenerativeHead(nn.Module):
    """Interface for a per-patch generative decoder.

    Both methods treat patches as an independent batch of vectors (cross-patch information is
    already baked into `cond` by the backbone), so internally we flatten (B, N, ...) -> (B*N, ...).

    Contract:
        loss(x0, cond)   -> scalar training objective.
        sample(cond)     -> reconstructed clean patches (B, N, patch_dim), from noise given cond.
    where x0:(B, N, patch_dim) clean patches, cond:(B, N, cond_dim) per-patch hidden states.
    """

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _ResMLPBlock(nn.Module):
    """Residual MLP block with additive conditioning injected at its input (pre-norm)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(F.gelu(self.fc1(self.norm(h) + c)))


class PatchDenoiser(nn.Module):
    """Small per-patch noise predictor: (noised patch, timestep t, patch hidden state) -> predicted noise.

    Operates on a flat batch of patches (M, patch_dim) where M = B*N. The timestep embedding and the
    conditioning hidden state are summed into one signal that is injected into every residual block
    (the same additive-injection idea as diffusion.py's ResBlock).
    """

    def __init__(self, patch_dim: int, cond_dim: int, hidden: int = 256, num_blocks: int = 3, time_emb_dim: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(patch_dim, hidden)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList(_ResMLPBlock(hidden) for _ in range(num_blocks))
        self.out_proj = nn.Linear(hidden, patch_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x_t:(M, patch_dim)  t:(M,)  cond:(M, cond_dim)
        h = self.in_proj(x_t)
        c = self.time_mlp(t) + self.cond_proj(cond)  # (M, hidden) conditioning signal
        for block in self.blocks:
            h = block(h, c)
        # (M, patch_dim): interpreted as predicted noise (diffusion head) or velocity (flow head).
        return self.out_proj(h)


class DiffusionPatchHead(PatchGenerativeHead):
    """Diffusion generative head. Reuses diffusion.py's schedule, q_sample, and reverse chain.

    Diffusion happens in pixel-patch space (patch_dim = C*p*p). Each patch draws its own timestep t
    during training. Reconstruction starts from *pure noise* (t_start = T) and relies entirely on the
    conditioning hidden state to recover the clean patch.
    """

    def __init__(self, patch_dim: int, cond_dim: int, timesteps: int = 1000,
                 hidden: int = 256, num_blocks: int = 3, time_emb_dim: int = 128):
        super().__init__()
        self.patch_dim = patch_dim
        self.diffusion = Diffusion(timesteps=timesteps)  # schedule buffers + q_sample + reverse step
        self.denoiser = PatchDenoiser(patch_dim, cond_dim, hidden, num_blocks, time_emb_dim)

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, p = x0.shape
        x0f = x0.reshape(b * n, p)
        condf = cond.reshape(b * n, cond.shape[-1])
        m = b * n
        # Per-patch independent timestep (M,), NOT shared across an image's patches.
        t = torch.randint(0, self.diffusion.timesteps, (m,), device=x0.device)
        noise = torch.randn_like(x0f)
        x_t = self.diffusion.q_sample(x0f, t, noise)         # closed-form forward noising
        predicted = self.denoiser(x_t, t, condf)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        b, n, d = cond.shape
        condf = cond.reshape(b * n, d)
        device = cond.device

        # Wrap the conditional denoiser into the 2-arg model(x, t) signature the reverse chain expects,
        # capturing the fixed condition. p_sample_step already takes `model` as a param, so no edit
        # to diffusion.py is needed.
        def model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.denoiser(x, t, condf)

        patches = self.diffusion.sample(model, (b * n, self.patch_dim), device=device)
        return patches.reshape(b, n, self.patch_dim)


class FlowMatchingPatchHead(PatchGenerativeHead):
    """Flow-matching (rectified-flow) generative head. Swappable with DiffusionPatchHead.

    Conditional flow matching on the straight-line path between noise and data, in pixel-patch space
    (patch_dim = C*p*p). With x0 = noise and x1 = clean patch:
        x_t = (1 - t) * x0 + t * x1,    t in [0, 1]
    the network predicts the velocity v(x_t, t, cond). The path's (constant) velocity is x1 - x0, so
    the objective is MSE(v, x1 - x0). Reconstruction integrates dx/dt = v with forward Euler from
    t=0 (pure noise) to t=1, conditioned on the per-patch hidden state.

    Reuses PatchDenoiser verbatim as the velocity network -- its (M,patch_dim)+(M,)+(M,cond_dim) ->
    (M,patch_dim) signature is exactly a velocity field; the output is just read as velocity instead
    of noise. Holds NO Diffusion schedule (flow matching needs no betas / q_sample / reverse chain),
    so its parameter count matches the diffusion head's for a capacity-matched comparison.

    time_scale: the SinusoidalTimeEmbedding inside PatchDenoiser was tuned for integer timesteps
    (~0..1000); feeding raw t in [0, 1] leaves most of its channels near-constant, crippling the time
    conditioning. We therefore scale t by a fixed constant (default 1000) before the time MLP, applied
    IDENTICALLY in loss and sample via `_t_embed_input`, so the two paths see the same representation.
    """

    def __init__(self, patch_dim: int, cond_dim: int, hidden: int = 256, num_blocks: int = 3,
                 time_emb_dim: int = 128, sample_steps: int = 50, time_scale: float = 1000.0):
        super().__init__()
        self.patch_dim = patch_dim
        self.sample_steps = sample_steps          # default number of Euler steps (overridable per call)
        self.time_scale = time_scale              # fixed constant; NOT learned, NOT tied to any T
        # Reused verbatim from the diffusion head; here its output is interpreted as a velocity.
        self.velocity_net = PatchDenoiser(patch_dim, cond_dim, hidden, num_blocks, time_emb_dim)

    def _t_embed_input(self, t: torch.Tensor) -> torch.Tensor:
        """Map continuous t in [0, 1] to the integer-range argument SinusoidalTimeEmbedding expects.

        Both loss() and sample() route t through here, so the train- and sample-time time
        representations are guaranteed identical (the load-bearing detail for FM to work).
        """
        return t * self.time_scale

    def loss(self, x0: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # NOTE: the interface arg named `x0` is the CLEAN target patch (PatchReconstructionModel
        # passes patchify(images)); in flow-matching terms that is x1. The noise endpoint is drawn
        # fresh below as `x0_noise`. Swapping these two would flip the velocity sign.
        b, n, p = x0.shape
        x1f = x0.reshape(b * n, p)                       # clean patches (x1)
        condf = cond.reshape(b * n, cond.shape[-1])
        m = b * n
        x0_noise = torch.randn_like(x1f)                 # noise endpoint (x0) ~ N(0, I)
        # Per-patch independent continuous time in [0, 1) (not shared across an image's patches),
        # mirroring DiffusionPatchHead's per-patch randint.
        t = torch.rand(m, device=x0.device)
        tb = t[:, None]                                  # (M, 1) to broadcast over patch_dim
        x_t = (1.0 - tb) * x0_noise + tb * x1f           # straight-line interpolant
        v = self.velocity_net(x_t, self._t_embed_input(t), condf)
        target = x1f - x0_noise                          # constant target velocity of this path
        return F.mse_loss(v, target)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """Euler-integrate dx/dt = v from t=0 (pure noise) to t=1, conditioned on cond.

        num_steps overrides self.sample_steps for this call (for NFE speed/quality sweeps);
        sample(cond) keeps the default so the swappable PatchGenerativeHead interface still holds.
        """
        b, n, d = cond.shape
        condf = cond.reshape(b * n, d)
        m = b * n
        device = cond.device
        steps = num_steps if num_steps is not None else self.sample_steps

        x = torch.randn(m, self.patch_dim, device=device)   # x0 at t=0
        dt = 1.0 / steps
        # Left-endpoint grid t = i*dt for i in 0..S-1: exactly S network evals (NFE = steps), never
        # feeds t=1 (excluded by training's U[0,1)), and S updates advance the state from t=0 to t=1.
        for i in range(steps):
            t = torch.full((m,), i * dt, device=device)
            v = self.velocity_net(x, self._t_embed_input(t), condf)
            x = x + dt * v                                   # forward Euler step
        x = x.clamp(-1.0, 1.0)                               # once at the end (matches the diffusion head)
        return x.reshape(b, n, self.patch_dim)


# --------------------------------------------------------------------------- #
# Full model: backbone (encoder/bottleneck) + per-patch generative head (decoder)
# --------------------------------------------------------------------------- #
class PatchReconstructionModel(nn.Module):
    """encode an image to per-patch hidden states, then reconstruct each patch from noise.

    The hidden state is the compression bottleneck under study; the head is the conditional
    generative decoder. `head_kind` selects the decoder ("diffusion" or "flow"); the backbone and
    head capacity (head_hidden / head_blocks) are shared, so the two kinds are capacity-matched for a
    fair comparison. Pass an explicit `head` to override with any custom PatchGenerativeHead.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        channels: int = 3,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        head: PatchGenerativeHead | None = None,
        head_kind: str = "diffusion",
        timesteps: int = 1000,
        head_hidden: int = 256,
        head_blocks: int = 3,
        flow_sample_steps: int = 50,
        flow_time_scale: float = 1000.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.channels = channels

        self.backbone = PatchTransformer(
            img_size=img_size, patch_size=patch_size, channels=channels,
            dim=dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
        )
        patch_dim = channels * patch_size * patch_size
        # An explicit `head=` always wins; otherwise `head_kind` selects the generative decoder.
        # Both kinds wrap one identically-shaped PatchDenoiser, so head param counts match.
        if head is not None:
            self.head = head
        elif head_kind == "diffusion":
            self.head = DiffusionPatchHead(
                patch_dim=patch_dim, cond_dim=dim, timesteps=timesteps,
                hidden=head_hidden, num_blocks=head_blocks,
            )
        elif head_kind == "flow":
            self.head = FlowMatchingPatchHead(
                patch_dim=patch_dim, cond_dim=dim,
                hidden=head_hidden, num_blocks=head_blocks,
                sample_steps=flow_sample_steps, time_scale=flow_time_scale,
            )
        else:
            raise ValueError(f"unknown head_kind {head_kind!r}; choose 'diffusion' or 'flow'")

    def loss(self, images: torch.Tensor) -> torch.Tensor:
        cond = self.backbone(images)                       # (B, N, D) hidden states
        x0 = patchify(images, self.patch_size)             # (B, N, C*p*p) clean target patches
        return self.head.loss(x0, cond)

    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        cond = self.backbone(images)
        patches = self.head.sample(cond)                   # (B, N, C*p*p) from noise, given cond
        recon = unpatchify(patches, self.patch_size, self.img_size, self.img_size, self.channels)
        return recon.clamp(-1.0, 1.0)


def _smoke() -> None:
    torch.manual_seed(0)
    b, c, h, w, p = 2, 3, 32, 32, 8
    x = torch.randn(b, c, h, w, requires_grad=True)
    model = PatchTransformer(img_size=h, patch_size=p, channels=c, dim=128, depth=4, num_heads=4)

    n_patches = (h // p) * (w // p)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"input    : {tuple(x.shape)}")
    print(f"config   : dim=128 depth=4 heads=4 mlp_ratio=4 patch_size={p}  -> N={n_patches} tokens")
    print(f"params   : {n_params:,}")

    out = model(x)
    print(f"output   : {tuple(out.shape)}  (B, N, D)")
    assert out.shape == (b, n_patches, 128), out.shape
    assert torch.isfinite(out).all(), "non-finite output"

    # Forward + backward must run end to end.
    loss = out.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "backward failed"
    grad_norm = sum(q.grad.norm().item() for q in model.parameters() if q.grad is not None)
    print(f"backward : loss={loss.item():.4f}  param-grad-norm-sum={grad_norm:.4f}")

    # Attention should actually mix patches (output tokens not all identical).
    spread = out[0].std(dim=0).mean().item()
    print(f"sanity   : per-token std across patches = {spread:.4f} (>0 -> patches differ)")
    assert spread > 1e-4, "patch tokens collapsed to identical values"
    print("smoke OK: image -> tokens -> transformer blocks -> per-patch hidden states, fwd+bwd pass")


def _smoke_full() -> None:
    print("\n=== full model (backbone + diffusion head) smoke ===")
    torch.manual_seed(0)
    b, c, h, w, p = 2, 3, 32, 32, 8
    x = (torch.rand(b, c, h, w) * 2 - 1).requires_grad_(True)  # in [-1, 1]
    model = PatchReconstructionModel(
        img_size=h, patch_size=p, channels=c, dim=128, depth=4, num_heads=4,
        timesteps=20, head_hidden=128, head_blocks=2,  # tiny T so the reverse chain is cheap on CPU
    )
    n_params = sum(q.numel() for q in model.parameters())
    print(f"input    : {tuple(x.shape)}  range=[{x.min():.3f}, {x.max():.3f}]")
    print(f"params   : {n_params:,} (backbone + head)")

    # forward + backward through the training loss
    loss = model.loss(x)
    print(f"loss     : {loss.item():.4f}")
    assert loss.dim() == 0 and torch.isfinite(loss), "loss must be a finite scalar"
    loss.backward()
    bb_grad = sum(q.grad.norm().item() for q in model.backbone.parameters() if q.grad is not None)
    hd_grad = sum(q.grad.norm().item() for q in model.head.parameters() if q.grad is not None)
    print(f"backward : backbone-grad-norm={bb_grad:.4f}  head-grad-norm={hd_grad:.4f}")
    assert bb_grad > 0 and hd_grad > 0, "both backbone and head must receive gradients"

    # reconstruct: encode -> sample patches from pure noise given cond -> unpatchify
    recon = model.reconstruct(x)
    print(f"recon    : {tuple(recon.shape)}  range=[{recon.min():.3f}, {recon.max():.3f}]")
    assert recon.shape == x.shape, recon.shape
    assert torch.isfinite(recon).all(), "non-finite reconstruction"
    assert recon.min() >= -1.0 and recon.max() <= 1.0, "reconstruction not clamped to [-1, 1]"
    print("smoke OK: image -> hidden states -> per-patch conditional diffusion -> reconstruction, fwd+bwd pass")


def _smoke_flow() -> None:
    print("\n=== full model (backbone + flow-matching head) smoke ===")
    torch.manual_seed(0)
    b, c, h, w, p = 2, 3, 32, 32, 8
    x = (torch.rand(b, c, h, w) * 2 - 1).requires_grad_(True)  # in [-1, 1]
    model = PatchReconstructionModel(
        img_size=h, patch_size=p, channels=c, dim=128, depth=4, num_heads=4,
        head_kind="flow", head_hidden=128, head_blocks=2, flow_sample_steps=8,  # tiny S -> cheap on CPU
    )
    n_params = sum(q.numel() for q in model.parameters())
    print(f"input    : {tuple(x.shape)}  range=[{x.min():.3f}, {x.max():.3f}]")
    print(f"params   : {n_params:,} (backbone + flow head)")

    # forward + backward through the FM training loss
    loss = model.loss(x)
    print(f"loss     : {loss.item():.4f}")
    assert loss.dim() == 0 and torch.isfinite(loss), "loss must be a finite scalar"
    loss.backward()
    bb_grad = sum(q.grad.norm().item() for q in model.backbone.parameters() if q.grad is not None)
    hd_grad = sum(q.grad.norm().item() for q in model.head.parameters() if q.grad is not None)
    print(f"backward : backbone-grad-norm={bb_grad:.4f}  head-grad-norm={hd_grad:.4f}")
    assert bb_grad > 0 and hd_grad > 0, "both backbone and head must receive gradients"

    # reconstruct: encode -> Euler-integrate from noise to t=1 given cond -> unpatchify
    recon = model.reconstruct(x)
    print(f"recon    : {tuple(recon.shape)}  range=[{recon.min():.3f}, {recon.max():.3f}]")
    assert recon.shape == x.shape, recon.shape
    assert torch.isfinite(recon).all(), "non-finite reconstruction"
    assert recon.min() >= -1.0 and recon.max() <= 1.0, "reconstruction not clamped to [-1, 1]"

    # ---- FM-specific guards (catch this head's silent failure modes) ----
    head = model.head
    # (1) interpolant convention: t=0 -> pure noise, t=1 -> clean patch (locks the x0/x1 sign).
    x1f = torch.randn(5, head.patch_dim)
    x0_noise = torch.randn_like(x1f)
    for t_val, expect in ((0.0, x0_noise), (1.0, x1f)):
        tb = torch.full((5, 1), t_val)
        assert torch.allclose((1.0 - tb) * x0_noise + tb * x1f, expect, atol=1e-6), \
            f"interpolant endpoint wrong at t={t_val}"
    # (2) flow head carries NO diffusion schedule.
    assert not hasattr(head, "diffusion"), "flow head must not carry a Diffusion schedule"
    # (3) time_scale restores the embedding's dynamic range (guards against dropping the scaling).
    emb = head.velocity_net.time_mlp[0]  # the SinusoidalTimeEmbedding
    t_lin = torch.linspace(0, 1, 16)
    raw_std = emb(t_lin).std(dim=0).mean().item()
    scaled_std = emb(head._t_embed_input(t_lin)).std(dim=0).mean().item()
    print(f"time-emb : raw-t std={raw_std:.4f}  scaled-t std={scaled_std:.4f}  (scaled must be >>)")
    assert scaled_std > 5 * raw_std, "time_scale is not restoring the time-embedding range"
    # (4) per-call num_steps override works and Euler is deterministic under a fixed seed.
    with torch.no_grad():
        cond = model.backbone(x)
    torch.manual_seed(7); s1 = head.sample(cond, num_steps=4)
    torch.manual_seed(7); s2 = head.sample(cond, num_steps=4)
    assert s1.shape == (b, cond.shape[1], head.patch_dim), s1.shape
    assert torch.allclose(s1, s2), "Euler sampling not deterministic under a fixed seed"
    # (5) capacity match: flow head and diffusion head have equal param counts (same PatchDenoiser).
    diff_model = PatchReconstructionModel(
        img_size=h, patch_size=p, channels=c, dim=128, depth=4, num_heads=4,
        head_kind="diffusion", head_hidden=128, head_blocks=2, timesteps=20,
    )
    fm_params = sum(q.numel() for q in model.head.parameters())
    df_params = sum(q.numel() for q in diff_model.head.parameters())
    print(f"capacity : flow-head params={fm_params:,}  diffusion-head params={df_params:,}  (must match)")
    assert fm_params == df_params, "head capacities differ -> unfair diffusion-vs-flow comparison"
    print("smoke OK: image -> hidden states -> per-patch flow matching -> reconstruction, fwd+bwd pass")


if __name__ == "__main__":
    _smoke()
    _smoke_full()
    _smoke_flow()
