"""Contextual-bandit distribution learning and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from phitd.cf import build_categorical_cf, build_quantile_cf, sample_frequencies
from phitd.style import algo_color, save_pdf, style_axes_panel

jax.config.update("jax_enable_x64", True)
_T975_DF1_TO_30: tuple[float, ...] = (
    12.70620473617471,
    4.30265272974946,
    3.18244630528371,
    2.7764451051978,
    2.57058183563637,
    2.446911846023004,
    2.364624251592874,
    2.305962926813099,
    2.262157158654392,
    2.228138851986496,
    2.200985160082949,
    2.178812829672518,
    2.160368652931024,
    2.144786687053816,
    2.131449546022662,
    2.119905299285527,
    2.109815577583358,
    2.100922040309607,
    2.093024054408617,
    2.085963447428391,
    2.079613844727773,
    2.073873067878009,
    2.068657610523972,
    2.063898561628096,
    2.059538552753294,
    2.055529438843488,
    2.051830516480699,
    2.048407141795935,
    2.045229642134926,
    2.042272456301236,
)


def _t975_critical(df: int) -> float:
    if df < 1:
        return 0.0
    if df <= len(_T975_DF1_TO_30):
        return float(_T975_DF1_TO_30[df - 1])
    return 1.96


def _mean_ci95_halfwidth(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """Half-width of 95% CI for the mean (Student t), NaN-aware."""
    n = int(stack.shape[axis])
    if n < 2:
        return np.zeros_like(np.nanmean(stack, axis=axis))
    s = np.nanstd(stack, axis=axis, ddof=1)
    return (_t975_critical(n - 1) * s / np.sqrt(n)).astype(np.float64)


def _huber(u: jax.Array, kappa: float) -> jax.Array:
    au = jnp.abs(u)
    return jnp.where(au <= kappa, 0.5 * u**2, kappa * (au - 0.5 * kappa))


def quantile_huber_loss_pairwise(
    q_preds: jax.Array, rewards: jax.Array, taus: jax.Array, kappa: float
) -> jax.Array:
    u = rewards[:, None] - q_preds
    hub = _huber(u, kappa)
    ind = (u < 0).astype(q_preds.dtype)
    return jnp.mean(jnp.sum(jnp.abs(taus - ind) * hub / kappa, axis=-1))


def gil_pelaez_cdf(
    phi_positive: jax.Array, t_pos: jax.Array, x_grid: jax.Array
) -> jax.Array:
    phi_positive = phi_positive - 1j * jnp.imag(phi_positive[0])
    x = x_grid[:, None]
    t = t_pos[None, :]
    integrand = jnp.imag(jnp.exp(-1j * t * x) * phi_positive) / t
    return 0.5 - jnp.trapezoid(integrand, t_pos, axis=1) / jnp.pi


def cramer_l2_sq_cdf(
    f_hat: jax.Array, f_true: jax.Array, x_grid: jax.Array
) -> jax.Array:
    return jnp.trapezoid(jnp.square(f_hat - f_true), x_grid)


_INVERSE_2PI = 1.0 / (2.0 * float(np.pi))


def cf_l2_sq_over_omega2(
    phi_hat: jax.Array, phi_t: jax.Array, omega_grid: jax.Array, omega_eps: float
) -> jax.Array:
    d = phi_hat - phi_t
    num = jnp.real(d) ** 2 + jnp.imag(d) ** 2
    den = jnp.maximum(jnp.square(jnp.abs(omega_grid)), jnp.float64(omega_eps**2))
    return jnp.trapezoid(num / den, omega_grid) * _INVERSE_2PI


def w1_cdf(f_hat: jax.Array, f_true: jax.Array, x_grid: jax.Array) -> jax.Array:
    return jnp.trapezoid(jnp.abs(f_hat - f_true), x_grid)


def empirical_cdf(sorted_samples: jax.Array, x_grid: jax.Array) -> jax.Array:
    n = sorted_samples.shape[0]
    idx = jnp.searchsorted(sorted_samples, x_grid, side="right")
    return idx.astype(jnp.float64) / jnp.maximum(n, 1)


def empirical_cf_from_samples(samples: jax.Array, t_grid: jax.Array) -> jax.Array:
    return jnp.mean(jnp.exp(1j * t_grid[:, None] * samples[None, :]), axis=1)


def project_delta_onto_c51_atoms(y: jax.Array, support: jax.Array) -> jax.Array:
    vmin, vmax = (support[0], support[-1])
    n = support.shape[0]
    dz = (vmax - vmin) / jnp.float32(n - 1)
    y_c = jnp.clip(y, vmin, vmax)
    b = (y_c - vmin) / dz
    lo = jnp.floor(b).astype(jnp.int32)
    hi = jnp.ceil(b).astype(jnp.int32)
    lo = jnp.clip(lo, 0, n - 1)
    hi = jnp.clip(hi, 0, n - 1)
    lm = hi.astype(jnp.float32) - b
    um = b - lo.astype(jnp.float32)
    same = lo == hi
    lm, um = (
        jnp.where(same, jnp.ones_like(lm), lm),
        jnp.where(same, jnp.zeros_like(um), um),
    )
    batch = jnp.arange(y.shape[0], dtype=jnp.int32)
    mat = jnp.zeros((y.shape[0], n), jnp.float32)
    mat = mat.at[batch, lo].add(lm)
    return mat.at[batch, hi].add(um)


def discrete_return_cdf(
    probs: jax.Array, atoms: jax.Array, x_grid: jax.Array
) -> jax.Array:
    return jnp.sum(
        probs[:, None] * (atoms[:, None] <= x_grid[None, :]).astype(jnp.float64), axis=0
    )


def discrete_return_cf(probs: jax.Array, atoms: jax.Array, t: jax.Array) -> jax.Array:
    return jnp.sum(
        probs[:, None]
        * jnp.exp(1j * atoms[:, None] * t[None, :].astype(jnp.complex128)),
        axis=0,
    )


def dirac_mixture_cf_batched(pi: jax.Array, mu: jax.Array, om: jax.Array) -> jax.Array:
    """Mixture-of-Diracs CF; pi, mu (B, M), om (B, N) → (B, N) complex."""
    return jnp.sum(
        pi[..., None, :] * jnp.exp(1j * om[..., None] * mu[..., None, :]), axis=-1
    )


def phi_dirac_cf_on_grid(pi: jax.Array, mu: jax.Array, t_grid: jax.Array) -> jax.Array:
    """π, μ over components (M,) and frequencies t_grid (T,) → φ(t) complex (T,)."""
    return jnp.sum(pi[:, None] * jnp.exp(1j * t_grid[None, :] * mu[:, None]), axis=0)


def mixture_dirac_cdf(x_grid: jax.Array, pi: jax.Array, mu: jax.Array) -> jax.Array:
    return jnp.sum(
        pi[:, None] * (mu[:, None] <= x_grid[None, :]).astype(jnp.float64), axis=0
    )


NUM_ACTIONS = 4
STATE_DIM = 5
GAUSS_SIGMA = 1.0
CAUCHY_PEAK_BASE, CAUCHY_PEAK_SCALE, CAUCHY_GAMMA = (2.0, 2.0, 0.5)
MOG_MU, MOG_SIGMA_FLOOR, MOG_SIGMA_SCALE = (3.0, 0.05, 0.4)
GAMMA_SHAPE = 2.0
GAMMA_LOG_THETA_BASE, GAMMA_LOG_THETA_SCALE = (-0.15, 0.28)


def _cauchy_d(x_state):
    return CAUCHY_PEAK_BASE + CAUCHY_PEAK_SCALE * jnp.abs(x_state[..., 1])


def _mog_sigma(x_state):
    return MOG_SIGMA_FLOOR + MOG_SIGMA_SCALE * jnp.abs(x_state[..., 2])


def _gamma_scale(x_state):
    return jnp.exp(GAMMA_LOG_THETA_BASE + GAMMA_LOG_THETA_SCALE * x_state[..., 3])


def _gauss_sample(rng, x_state):
    mu = x_state[..., 0]
    return mu + GAUSS_SIGMA * jax.random.normal(rng, mu.shape, dtype=x_state.dtype)


def _gauss_phi(t, x_state):
    mu = x_state[0]
    return jnp.exp(1j * t * mu - 0.5 * GAUSS_SIGMA**2 * t**2)


def _cauchy_sample(rng, x_state):
    rk1, rk2 = jax.random.split(rng)
    pick = jax.random.bernoulli(rk1, 0.5, x_state.shape[:-1])
    u = jax.random.uniform(rk2, x_state.shape[:-1], dtype=x_state.dtype)
    z, d = (jnp.tan(jnp.pi * (u - 0.5)), _cauchy_d(x_state))
    return jnp.where(pick, -d + CAUCHY_GAMMA * z, d + CAUCHY_GAMMA * z)


def _cauchy_phi(t, x_state):
    d, at = (_cauchy_d(x_state), jnp.abs(t))
    return 0.5 * (
        jnp.exp(-1j * t * d - CAUCHY_GAMMA * at)
        + jnp.exp(1j * t * d - CAUCHY_GAMMA * at)
    )


def _mog_sample(rng, x_state):
    rk1, rk2 = jax.random.split(rng)
    pick = jax.random.bernoulli(rk1, 0.5, x_state.shape[:-1])
    eps = jax.random.normal(rk2, x_state.shape[:-1], dtype=x_state.dtype)
    sig = _mog_sigma(x_state)
    return jnp.where(pick, -MOG_MU, MOG_MU) + sig * eps


def _mog_phi(t, x_state):
    sig = _mog_sigma(x_state)
    return (jnp.exp(-0.5 * sig**2 * t**2) * jnp.cos(MOG_MU * t)).astype(jnp.complex128)


def _gamma_sample(rng, x_state):
    th = _gamma_scale(x_state)
    g = jax.random.gamma(rng, GAMMA_SHAPE, x_state.shape[:-1], dtype=x_state.dtype)
    return th * g


def _gamma_phi(t, x_state):
    th = _gamma_scale(x_state)
    z = (1.0 - 1j * th * t).astype(jnp.complex128)
    return jnp.power(z, -jnp.float64(GAMMA_SHAPE))


def _gauss_cdf(x_grid, x_state):
    mu = x_state[0]
    return 0.5 * (
        1.0 + jax.scipy.special.erf((x_grid - mu) / (GAUSS_SIGMA * jnp.sqrt(2.0)))
    )


def _cauchy_cdf(x_grid, x_state):
    d = _cauchy_d(x_state)
    return 0.5 * (
        jnp.arctan((x_grid + d) / CAUCHY_GAMMA) / jnp.pi
        + 0.5
        + jnp.arctan((x_grid - d) / CAUCHY_GAMMA) / jnp.pi
        + 0.5
    )


def _mog_cdf(x_grid, x_state):
    sig = _mog_sigma(x_state)
    s2 = sig * jnp.sqrt(2.0)
    f1 = 0.5 * (1.0 + jax.scipy.special.erf((x_grid + MOG_MU) / s2))
    f2 = 0.5 * (1.0 + jax.scipy.special.erf((x_grid - MOG_MU) / s2))
    return 0.5 * (f1 + f2)


def _gamma_cdf(x_grid, x_state):
    th = _gamma_scale(x_state)
    a = jnp.float64(GAMMA_SHAPE)
    rat = jnp.maximum(x_grid, 0.0) / jnp.maximum(th, jnp.finfo(jnp.float64).tiny)
    return jnp.where(x_grid > 0, jax.scipy.special.gammainc(a, rat), 0.0)


def _gauss_pdf(x_grid, x_state):
    mu = x_state[0]
    return jnp.exp(-0.5 * ((x_grid - mu) / GAUSS_SIGMA) ** 2) / (
        GAUSS_SIGMA * jnp.sqrt(2.0 * jnp.pi)
    )


def _cauchy_pdf(x_grid, x_state):
    d = _cauchy_d(x_state)
    p1 = 1.0 / (jnp.pi * CAUCHY_GAMMA * (1.0 + ((x_grid + d) / CAUCHY_GAMMA) ** 2))
    p2 = 1.0 / (jnp.pi * CAUCHY_GAMMA * (1.0 + ((x_grid - d) / CAUCHY_GAMMA) ** 2))
    return 0.5 * (p1 + p2)


def _mog_pdf(x_grid, x_state):
    sig = _mog_sigma(x_state)
    p1 = jnp.exp(-0.5 * ((x_grid + MOG_MU) / sig) ** 2) / (sig * jnp.sqrt(2.0 * jnp.pi))
    p2 = jnp.exp(-0.5 * ((x_grid - MOG_MU) / sig) ** 2) / (sig * jnp.sqrt(2.0 * jnp.pi))
    return 0.5 * (p1 + p2)


def _gamma_pdf(x_grid, x_state):
    th = _gamma_scale(x_state)
    a = jnp.float64(GAMMA_SHAPE)
    x = jnp.maximum(x_grid, jnp.finfo(jnp.float64).tiny)
    logp = (
        (a - 1.0) * jnp.log(x) - x / th - jax.scipy.special.gammaln(a) - a * jnp.log(th)
    )
    return jnp.where(x_grid > 0, jnp.exp(logp), 0.0)


@dataclass(frozen=True)
class ActionDist:
    name: str
    short: str
    sample: Callable
    phi: Callable
    cdf: Callable
    pdf: Callable
    plot_x_min: float
    plot_x_max: float


ACTION_DISTS = (
    ActionDist(
        "Gaussian (baseline)",
        "Gauss",
        _gauss_sample,
        _gauss_phi,
        _gauss_cdf,
        _gauss_pdf,
        -6.0,
        6.0,
    ),
    ActionDist(
        "Cauchy mixture (heavy-tailed)",
        "Cauchy",
        _cauchy_sample,
        _cauchy_phi,
        _cauchy_cdf,
        _cauchy_pdf,
        -12.0,
        12.0,
    ),
    ActionDist(
        "MoG (sharp / bimodal)",
        "MoG",
        _mog_sample,
        _mog_phi,
        _mog_cdf,
        _mog_pdf,
        -6.0,
        6.0,
    ),
    ActionDist(
        "Gamma (skewed positive)",
        "Gam",
        _gamma_sample,
        _gamma_phi,
        _gamma_cdf,
        _gamma_pdf,
        -0.5,
        8.0,
    ),
)
DATASET_SEED_OFFSET = 1001
_PANEL_ANCHOR_PRNG_SEED = 42424242


def _panel_anchor_prng_key() -> jax.Array:
    """Lazy key — avoids running JAX on GPU at import time (spawn workers import this module)."""
    return jax.random.PRNGKey(_PANEL_ANCHOR_PRNG_SEED)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    num_seeds: int = 1
    state_dim: int = STATE_DIM
    dataset_size: int = 500000
    dataset_seed_offset: int = DATASET_SEED_OFFSET
    total_steps: int = 20000
    parseval_every: int = 1000
    batch_size: int = 1024
    num_tau: int = 51
    num_atoms: int = 51
    num_dirac_components: int = 51
    num_omega: int = 32
    gamma: float = 0.9
    immediate_reward_std: float = 0.2
    omega_max: float = 15.0
    huber_kappa: float = 1.0
    lr: float = 0.001
    max_grad_norm: float = 10.0
    omega_clip: float = 15.0
    omega_laplace_scale: float = 0.8
    cf_loss_omega_eps: float = 0.001
    aux_mean_loss_weight: float = 0.0
    c51_v_min: float = -25.0
    c51_v_max: float = 25.0
    hidden_dim: int = 256
    trunk_layers: int = 2
    eval_num_t: int = 201
    eval_x_min: float = -15.0
    eval_x_max: float = 15.0
    eval_num_x: int = 301
    gp_t_delta: float = 0.0001
    gp_num_t: int = 2001
    close_to_theory: bool = False
    omega_min_pareto: float = 0.001
    phi_qt_same_arch_as_qtd: bool = True
    phi_qt_num_quantiles: int | None = None
    phi_qt_aux_qtd_huber_weight: float = 0.0


def _default_distribution_yaml_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "bandit.yaml"


def load_da_config_yaml(path: Path | None, overrides: dict) -> Config:
    from omegaconf import OmegaConf

    p = path or _default_distribution_yaml_path()
    raw = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown bandit settings: {sorted(unknown)}")
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**raw)


def fixed_qr_tau_grid(n: int, dtype=jnp.float32):
    idx = jnp.arange(1, n + 1, dtype=dtype)
    return (idx - jnp.float32(0.5)) / jnp.float32(n)


def c51_atoms(cfg: Config):
    return jnp.linspace(
        jnp.float32(cfg.c51_v_min),
        jnp.float32(cfg.c51_v_max),
        cfg.num_atoms,
        dtype=jnp.float32,
    )


def ctd_atoms(cfg: Config):
    return c51_atoms(cfg)


def _dense_ln_relu(x: jax.Array, hidden_dim: int, name: str) -> jax.Array:
    x = nn.Dense(hidden_dim, name=f"{name}_dense")(x)
    x = nn.LayerNorm(name=f"{name}_ln")(x)
    return nn.relu(x)


class QTDNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_quantiles: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        return nn.Dense(self.num_quantiles, name="qt")(h)


class CTDNet(nn.Module):
    hidden_dim: int
    trunk_layers: int
    num_atoms: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        return nn.Dense(self.num_atoms, name="at")(h)


class PhiDiracNet(nn.Module):
    """Mixture of Diracs (π, μ) per action — same head layout as MoG without σ."""

    hidden_dim: int
    trunk_layers: int
    num_components: int

    @nn.compact
    def __call__(self, x, a_onehot):
        h = jnp.concatenate([x, a_onehot], -1)
        for i in range(self.trunk_layers):
            h = _dense_ln_relu(h, self.hidden_dim, f"t{i}")
        pi = jax.nn.softmax(nn.Dense(self.num_components, name="pi")(h), -1)
        mu = nn.Dense(self.num_components, name="mu")(h)
        return (pi, mu)


def _opt(lr: float, clip: float):
    return optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))


def _build_qtd_net(cfg: Config, *, num_quantiles: int | None = None) -> QTDNet:
    """Same ``QTDNet`` spec as QTD; ``num_quantiles`` overrides ``cfg.num_tau`` when set."""
    nq = int(cfg.num_tau if num_quantiles is None else num_quantiles)
    return QTDNet(cfg.hidden_dim, cfg.trunk_layers, nq)


def _phi_qt_output_dim(cfg: Config) -> int:
    """Output width of the φTD Quant head (matches ``num_tau`` when ``phi_qt_same_arch_as_qtd``)."""
    if cfg.phi_qt_same_arch_as_qtd:
        return int(cfg.num_tau)
    return int(
        cfg.phi_qt_num_quantiles
        if cfg.phi_qt_num_quantiles is not None
        else cfg.num_tau
    )


def create_train_states(rng: jax.Array, cfg: Config):
    rctd, rqtd, rphid, rphic, rphiq = jax.random.split(rng, 5)
    ctd = CTDNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_atoms)
    qtd = _build_qtd_net(cfg)
    phi_d = PhiDiracNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_dirac_components)
    phi_cat = CTDNet(cfg.hidden_dim, cfg.trunk_layers, cfg.num_atoms)
    if cfg.phi_qt_same_arch_as_qtd:
        phi_qt = _build_qtd_net(cfg)
    else:
        n_phi = (
            cfg.phi_qt_num_quantiles
            if cfg.phi_qt_num_quantiles is not None
            else cfg.num_tau
        )
        phi_qt = _build_qtd_net(cfg, num_quantiles=n_phi)
    x0, a0 = (
        jnp.zeros((2, cfg.state_dim), jnp.float32),
        jnp.zeros((2, NUM_ACTIONS), jnp.float32),
    )
    tx = _opt(cfg.lr, cfg.max_grad_norm)
    v5, vq, vd, vc, vqt = (
        ctd.init(rctd, x0, a0),
        qtd.init(rqtd, x0, a0),
        phi_d.init(rphid, x0, a0),
        phi_cat.init(rphic, x0, a0),
        phi_qt.init(rphiq, x0, a0),
    )
    return (
        train_state.TrainState.create(apply_fn=ctd.apply, params=v5["params"], tx=tx),
        train_state.TrainState.create(apply_fn=qtd.apply, params=vq["params"], tx=tx),
        train_state.TrainState.create(apply_fn=phi_d.apply, params=vd["params"], tx=tx),
        train_state.TrainState.create(
            apply_fn=phi_cat.apply, params=vc["params"], tx=tx
        ),
        train_state.TrainState.create(
            apply_fn=phi_qt.apply, params=vqt["params"], tx=tx
        ),
        ctd,
        qtd,
        phi_d,
        phi_cat,
        phi_qt,
    )


def build_dataset(rng: jax.Array, cfg: Config):
    rx, ra, rr, rz = jax.random.split(rng, 4)
    n, d = (cfg.dataset_size, cfg.state_dim)
    x = jax.random.normal(rx, (n, d), jnp.float32)
    a = jax.random.randint(ra, (n,), 0, NUM_ACTIONS)
    r = jax.random.normal(rr, (n,), jnp.float32) * jnp.float32(cfg.immediate_reward_std)
    sk = jax.random.split(rz, NUM_ACTIONS)
    zn = jnp.stack(
        [
            ACTION_DISTS[k].sample(sk[k], x).astype(jnp.float32)
            for k in range(NUM_ACTIONS)
        ]
    )
    return (x, a, r, zn[a, jnp.arange(n)])


def make_train_chunk(
    cfg: Config,
    ctd: CTDNet,
    qtd: QTDNet,
    phi_d: PhiDiracNet,
    phi_cat: CTDNet,
    phi_qt: QTDNet,
    ds,
):
    xa, aa, ra, zn = ds
    n, aux_w = (xa.shape[0], jnp.float32(cfg.aux_mean_loss_weight))
    g, atoms = (jnp.float32(cfg.gamma), ctd_atoms(cfg))
    m_phi = _phi_qt_output_dim(cfg)
    tauf_phi = jnp.broadcast_to(
        fixed_qr_tau_grid(m_phi)[None, :], (cfg.batch_size, m_phi)
    )
    qtd_align_w = jnp.float32(cfg.phi_qt_aux_qtd_huber_weight)

    def step(c, _):
        sctd, sqtd, sphid, sphic, sphiq, rng = c
        rng, kb, kt, ko = jax.random.split(rng, 4)
        idx = jax.random.randint(kb, (cfg.batch_size,), 0, n)
        xb, ab, rb, znb = (xa[idx], aa[idx], ra[idx], zn[idx])
        oh = jax.nn.one_hot(ab, NUM_ACTIONS, dtype=jnp.float32)
        tauf = jnp.broadcast_to(
            fixed_qr_tau_grid(cfg.num_tau)[None, :], (cfg.batch_size, cfg.num_tau)
        )
        if cfg.close_to_theory:
            om = sample_frequencies(
                ko,
                cfg.batch_size * cfg.num_omega,
                float(cfg.omega_clip),
                scale=None,
                distribution="pareto_1",
                omega_min=float(cfg.omega_min_pareto),
            ).reshape(cfg.batch_size, cfg.num_omega)
        else:
            om = sample_frequencies(
                ko,
                cfg.batch_size * cfg.num_omega,
                float(cfg.omega_clip),
                scale=float(cfg.omega_laplace_scale),
                distribution="half_laplacian",
            ).reshape(cfg.batch_size, cfg.num_omega)
        tgt = rb + g * znb
        target_cf = jnp.exp(
            1j * om.astype(jnp.complex64) * tgt[:, None].astype(jnp.complex64)
        )
        if cfg.close_to_theory:

            def cf_phi_mean(residual: jnp.ndarray) -> jnp.ndarray:
                e2 = jnp.real(residual) ** 2 + jnp.imag(residual) ** 2
                return 0.5 * jnp.mean(e2) * _INVERSE_2PI
        else:
            w2 = jnp.maximum(jnp.square(om), jnp.float32(cfg.cf_loss_omega_eps**2))

            def cf_phi_mean(residual: jnp.ndarray) -> jnp.ndarray:
                e2 = jnp.real(residual) ** 2 + jnp.imag(residual) ** 2
                return 0.5 * jnp.mean(e2 / w2) * _INVERSE_2PI

        def lctd(pp):
            lg = jax.nn.log_softmax(ctd.apply({"params": pp}, xb, oh))
            tp = project_delta_onto_c51_atoms(tgt, atoms)
            return -jnp.mean(jnp.sum(tp * lg, -1))

        def lqtd(pp):
            return quantile_huber_loss_pairwise(
                qtd.apply({"params": pp}, xb, oh), tgt, tauf, cfg.huber_kappa
            )

        def lphid(pp):
            pi, mu = phi_d.apply({"params": pp}, xb, oh)
            pred = dirac_mixture_cf_batched(pi, mu, om)
            d = pred - target_cf
            mp = jnp.sum(pi * mu, axis=-1)
            return cf_phi_mean(d) + aux_w * jnp.mean(_huber(mp - tgt, cfg.huber_kappa))

        def lphic(pp):
            logits = phi_cat.apply({"params": pp}, xb, oh)
            probs = jax.nn.softmax(logits, axis=-1)

            def row_cat(p, o):
                return build_categorical_cf(p, atoms, o)

            pred = jax.vmap(row_cat)(probs, om)
            d = pred - target_cf
            mexp = jnp.sum(probs * atoms, axis=-1)
            return cf_phi_mean(d) + aux_w * jnp.mean(
                _huber(mexp - tgt, cfg.huber_kappa)
            )

        def lphiq(pp):
            qv = phi_qt.apply({"params": pp}, xb, oh)

            def row_qt(q, o):
                return build_quantile_cf(q, o)

            pred = jax.vmap(row_qt)(qv, om)
            d = pred - target_cf
            mq = jnp.mean(qv, axis=-1)
            loss_cf = cf_phi_mean(d) + aux_w * jnp.mean(
                _huber(mq - tgt, cfg.huber_kappa)
            )
            hub_qtd = quantile_huber_loss_pairwise(qv, tgt, tauf_phi, cfg.huber_kappa)
            return loss_cf + qtd_align_w * hub_qtd

        v5, g5 = jax.value_and_grad(lctd)(sctd.params)
        vq, gq = jax.value_and_grad(lqtd)(sqtd.params)
        vd, gd = jax.value_and_grad(lphid)(sphid.params)
        vc, gc = jax.value_and_grad(lphic)(sphic.params)
        vqt, gqt = jax.value_and_grad(lphiq)(sphiq.params)
        return (
            (
                sctd.apply_gradients(grads=g5),
                sqtd.apply_gradients(grads=gq),
                sphid.apply_gradients(grads=gd),
                sphic.apply_gradients(grads=gc),
                sphiq.apply_gradients(grads=gqt),
                rng,
            ),
            (v5, vq, vd, vc, vqt),
        )

    scan_len = int(cfg.parseval_every)

    @jax.jit
    def run(sctd, sqtd, sphid, sphic, sphiq, rng):
        (sctd, sqtd, sphid, sphic, sphiq, rng), loss = jax.lax.scan(
            step, (sctd, sqtd, sphid, sphic, sphiq, rng), None, scan_len
        )
        return (sctd, sqtd, sphid, sphic, sphiq, rng, loss)

    return run


def _make_eval(
    cfg: Config,
    ctd: CTDNet,
    qtd: QTDNet,
    phi_d: PhiDiracNet,
    phi_cat: CTDNet,
    phi_qt: QTDNet,
):
    wm = cfg.omega_max
    te = jnp.linspace(-wm, wm, cfg.eval_num_t, jnp.float64)
    xe = jnp.linspace(cfg.eval_x_min, cfg.eval_x_max, cfg.eval_num_x, jnp.float64)
    tp = jnp.linspace(cfg.gp_t_delta, wm, cfg.gp_num_t, jnp.float64)
    g64, rs64 = (jnp.float64(cfg.gamma), jnp.float64(cfg.immediate_reward_std))
    atoms = ctd_atoms(cfg)
    phis = [d.phi for d in ACTION_DISTS]

    def one(p5, pq, pd, pc, pqt, xs, k):
        oh = jax.nn.one_hot(jnp.array(k), NUM_ACTIONS, dtype=jnp.float32)
        x1, a1 = (xs[None, :].astype(jnp.float32), oh[None, :])
        lc = ctd.apply({"params": p5}, x1, a1)[0]
        ctd_probs = jax.nn.softmax(lc)
        F5 = discrete_return_cdf(ctd_probs, atoms, xe)
        phi5 = discrete_return_cf(ctd_probs, atoms, te)
        qq = qtd.apply({"params": pq}, x1, a1)[0].astype(jnp.float64)
        Fq = empirical_cdf(jnp.sort(qq), xe)
        phiq = empirical_cf_from_samples(
            qq.astype(jnp.complex128), te.astype(jnp.complex128)
        )
        piv, mv = phi_d.apply({"params": pd}, x1, a1)
        pmv, mmv = (piv[0], mv[0])
        phid = phi_dirac_cf_on_grid(pmv, mmv, te.astype(jnp.complex128))
        Fd = mixture_dirac_cdf(xe, pmv, mmv)
        lc2 = phi_cat.apply({"params": pc}, x1, a1)[0]
        pc2 = jax.nn.softmax(lc2)
        Fpc = discrete_return_cdf(pc2, atoms, xe)
        phipc = discrete_return_cf(pc2, atoms, te)
        qq2 = phi_qt.apply({"params": pqt}, x1, a1)[0].astype(jnp.float64)
        Fq2 = empirical_cdf(jnp.sort(qq2), xe)
        phiq2 = empirical_cf_from_samples(
            qq2.astype(jnp.complex128), te.astype(jnp.complex128)
        )
        pr = jnp.exp(-0.5 * rs64**2 * te**2)
        pz = jax.lax.switch(k, phis, g64 * te, xs.astype(jnp.float64))
        pt = pr * pz
        prp = jnp.exp(-0.5 * rs64**2 * tp**2)
        pzp = jax.lax.switch(k, phis, g64 * tp, xs.astype(jnp.float64))
        ptp = prp * pzp
        Ft = gil_pelaez_cdf(ptp, tp, xe)
        pdf_t = jnp.clip(jnp.gradient(Ft, xe), 0, None)
        pdf5 = jnp.clip(jnp.gradient(F5, xe), 0, None)
        pdfq = jnp.clip(jnp.gradient(Fq, xe), 0, None)
        pdfd = jnp.clip(jnp.gradient(Fd, xe), 0, None)
        pdfpc = jnp.clip(jnp.gradient(Fpc, xe), 0, None)
        pdfq2 = jnp.clip(jnp.gradient(Fq2, xe), 0, None)
        cf_l2_ctd = cf_l2_sq_over_omega2(phi5, pt, te, cfg.gp_t_delta)
        cf_l2_qtd = cf_l2_sq_over_omega2(phiq, pt, te, cfg.gp_t_delta)
        cf_l2_phi_dirac = cf_l2_sq_over_omega2(phid, pt, te, cfg.gp_t_delta)
        cf_l2_phi_cat = cf_l2_sq_over_omega2(phipc, pt, te, cfg.gp_t_delta)
        cf_l2_phi_qt = cf_l2_sq_over_omega2(phiq2, pt, te, cfg.gp_t_delta)
        out = {
            "w1_ctd": w1_cdf(F5, Ft, xe),
            "w1_qtd": w1_cdf(Fq, Ft, xe),
            "w1_dirac": w1_cdf(Fd, Ft, xe),
            "w1_phi_cat": w1_cdf(Fpc, Ft, xe),
            "w1_phi_qt": w1_cdf(Fq2, Ft, xe),
            "cramer_l2_ctd": cramer_l2_sq_cdf(F5, Ft, xe),
            "cramer_l2_qtd": cramer_l2_sq_cdf(Fq, Ft, xe),
            "cramer_l2_dirac": cramer_l2_sq_cdf(Fd, Ft, xe),
            "cramer_l2_phi_cat": cramer_l2_sq_cdf(Fpc, Ft, xe),
            "cramer_l2_phi_qt": cramer_l2_sq_cdf(Fq2, Ft, xe),
            "cf_l2_w_ctd": cf_l2_ctd,
            "cf_l2_w_qtd": cf_l2_qtd,
            "cf_l2_w_dirac": cf_l2_phi_dirac,
            "cf_l2_w_phi_cat": cf_l2_phi_cat,
            "cf_l2_w_phi_qt": cf_l2_phi_qt,
            "phi_ctd": phi5,
            "phi_qtd": phiq,
            "phi_dirac": phid,
            "phi_phi_cat": phipc,
            "phi_phi_qt": phiq2,
            "phi_true": pt,
            "F_ctd": F5,
            "F_qtd": Fq,
            "F_dirac": Fd,
            "F_phi_cat": Fpc,
            "F_phi_qt": Fq2,
            "F_true": Ft,
            "pdf_true": pdf_t,
            "pdf_ctd": pdf5,
            "pdf_qtd": pdfq,
            "pdf_dirac": pdfd,
            "pdf_phi_cat": pdfpc,
            "pdf_phi_qt": pdfq2,
        }
        return out

    @jax.jit
    def ev(p5, pq, pd, pc, pqt, xs):
        P = [one(p5, pq, pd, pc, pqt, xs, kk) for kk in range(NUM_ACTIONS)]
        return {k: jnp.stack([p[k] for p in P]) for k in P[0]}

    return (ev, te, xe)


PLOT_COLORS = {
    "truth": "#0A0A0A",
    "qtd": algo_color("qtd"),
    "ctd": algo_color("ctd"),
    "dirac": algo_color("phitd_fm"),
    "phi_cat": algo_color("phitd_fcm"),
    "phi_qt": algo_color("phitd_fqm"),
}
PLOT_LW = {
    "truth": 2.15,
    "qtd": 1.08,
    "ctd": 1.12,
    "dirac": 1.14,
    "phi_cat": 1.14,
    "phi_qt": 1.14,
}


def _plot_minimal_style(ax) -> None:
    style_axes_panel(ax)


def _plot_save_fig(fig, path: str, *, png_dpi: int = 300) -> None:
    save_pdf(fig, path)


def _plot_smooth_curve(y: np.ndarray, passes: int = 2) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 5 or passes <= 0:
        return arr
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    out = arr
    for _ in range(passes):
        padded = np.pad(out, (2, 2), mode="edge")
        out = np.convolve(padded, kernel, mode="valid")
    return out


def _plot_resolve_panel_key(
    data: dict[str, np.ndarray], candidates: tuple[str, ...]
) -> str | None:
    for key in candidates:
        if key in data:
            return key
    return None


def _plot_panels(
    mean_p: dict[str, np.ndarray],
    ci_p: dict[str, np.ndarray],
    te: np.ndarray,
    xe: np.ndarray,
    cf_hw: float,
    path: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    panel_colors = dict(PLOT_COLORS)
    cf_series = (
        (("panel_phi_true",), "truth", "Truth"),
        (("panel_phi_qtd",), "qtd", "QTD"),
        (("panel_phi_ctd",), "ctd", "CTD"),
        (("panel_phi_dirac",), "dirac", "$\\varphi$TD Dir"),
        (("panel_phi_phi_cat",), "phi_cat", "$\\varphi$TD Cat"),
        (("panel_phi_phi_qt",), "phi_qt", "$\\varphi$TD Quant"),
    )
    pdf_series = (
        (("panel_pdf_true",), "truth", "Truth"),
        (("panel_pdf_qtd",), "qtd", "QTD"),
        (("panel_pdf_ctd",), "ctd", "CTD"),
        (("panel_pdf_dirac",), "dirac", "$\\varphi$TD Dir"),
        (("panel_pdf_phi_cat",), "phi_cat", "$\\varphi$TD Cat"),
        (("panel_pdf_phi_qt",), "phi_qt", "$\\varphi$TD Quant"),
    )
    cdf_series = (
        (("panel_F_true",), "truth", "Truth"),
        (("panel_F_qtd",), "qtd", "QTD"),
        (("panel_F_ctd",), "ctd", "CTD"),
        (("panel_F_dirac",), "dirac", "$\\varphi$TD Dir"),
        (("panel_F_phi_cat",), "phi_cat", "$\\varphi$TD Cat"),
        (("panel_F_phi_qt",), "phi_qt", "$\\varphi$TD Quant"),
    )
    fig, axes = plt.subplots(
        len(ACTION_DISTS), 4, figsize=(13.5, 2.45 * len(ACTION_DISTS))
    )
    legend = None
    for row, dist in enumerate(ACTION_DISTS):
        axd, axcf, axp, axc = axes[row]
        xmask = (xe >= dist.plot_x_min) & (xe <= dist.plot_x_max)
        if "panel_pdf_true" in mean_p:
            y_truth = np.asarray(mean_p["panel_pdf_true"][row], dtype=np.float64)
            axd.plot(
                xe[xmask],
                y_truth[xmask],
                color=panel_colors["truth"],
                lw=PLOT_LW["truth"],
            )
        _plot_minimal_style(axd)
        axd.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axd.set_title("Density")
        tmask = np.abs(te) <= cf_hw
        for candidates, ckey, label in cf_series:
            pkey = _plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            y = np.real(mean_p[pkey][row])
            ys = np.real(ci_p.get(pkey, np.zeros_like(mean_p[pkey]))[row])
            axcf.fill_between(
                te[tmask],
                (y - ys)[tmask],
                (y + ys)[tmask],
                color=panel_colors[ckey],
                alpha=0.2,
                lw=0,
            )
            axcf.plot(
                te[tmask],
                y[tmask],
                color=panel_colors[ckey],
                lw=PLOT_LW[ckey],
                label=label,
            )
        _plot_minimal_style(axcf)
        axcf.set_xlim(-cf_hw, cf_hw)
        if row == 0:
            axcf.set_title("Re $\\varphi$")
        for candidates, ckey, label in pdf_series:
            pkey = _plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            if ckey in ("ctd", "phi_cat"):
                cdf_candidates = (
                    ("panel_F_ctd",) if ckey == "ctd" else ("panel_F_phi_cat",)
                )
                cdf_key = _plot_resolve_panel_key(mean_p, cdf_candidates)
                if cdf_key is None:
                    continue
                cdf_vals = np.asarray(mean_p[cdf_key][row], dtype=np.float64)
                jumps = np.diff(np.concatenate(([0.0], cdf_vals)))
                jumps = np.clip(jumps, 0.0, None)
                jump_se = np.zeros_like(jumps)
                if cdf_key in ci_p:
                    cdf_se = np.asarray(ci_p[cdf_key][row], dtype=np.float64)
                    jump_se = np.clip(
                        np.diff(np.concatenate(([0.0], cdf_se))), 0.0, None
                    )
                atom_x = xe
                keep = (
                    (atom_x >= dist.plot_x_min)
                    & (atom_x <= dist.plot_x_max)
                    & (jumps > 1e-06)
                )
                if np.any(keep):
                    segs = [
                        ((x, 0.0), (x, h)) for x, h in zip(atom_x[keep], jumps[keep])
                    ]
                    axp.add_collection(
                        LineCollection(
                            segs,
                            colors=panel_colors[ckey],
                            linewidths=PLOT_LW[ckey],
                            alpha=0.95,
                        )
                    )
                    if np.any(jump_se[keep] > 0):
                        axp.errorbar(
                            atom_x[keep],
                            jumps[keep],
                            yerr=jump_se[keep],
                            fmt="none",
                            ecolor=panel_colors[ckey],
                            elinewidth=0.8,
                            alpha=0.35,
                            capsize=0,
                        )
                    axp.plot(
                        [], [], color=panel_colors[ckey], lw=PLOT_LW[ckey], label=label
                    )
                continue
            if ckey == "truth":
                y = np.asarray(mean_p[pkey][row], dtype=np.float64)
                ys = np.asarray(
                    ci_p.get(pkey, np.zeros_like(mean_p[pkey]))[row], dtype=np.float64
                )
            else:
                y = _plot_smooth_curve(mean_p[pkey][row])
                ys = _plot_smooth_curve(
                    ci_p.get(pkey, np.zeros_like(mean_p[pkey]))[row], passes=1
                )
            axp.fill_between(
                xe[xmask],
                (y - ys)[xmask],
                (y + ys)[xmask],
                color=panel_colors[ckey],
                alpha=0.2,
                lw=0,
            )
            axp.plot(
                xe[xmask],
                y[xmask],
                color=panel_colors[ckey],
                lw=PLOT_LW[ckey],
                label=label,
            )
        _plot_minimal_style(axp)
        axp.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axp.set_title("PDF")
        for candidates, ckey, label in cdf_series:
            pkey = _plot_resolve_panel_key(mean_p, candidates)
            if pkey is None:
                continue
            y = mean_p[pkey][row]
            ys = ci_p.get(pkey, np.zeros_like(mean_p[pkey]))[row]
            axc.fill_between(
                xe[xmask],
                (y - ys)[xmask],
                (y + ys)[xmask],
                color=panel_colors[ckey],
                alpha=0.2,
                lw=0,
                step="post" if ckey in ("ctd", "phi_cat") else None,
            )
            if ckey in ("ctd", "phi_cat"):
                axc.step(
                    xe[xmask],
                    y[xmask],
                    where="post",
                    color=panel_colors[ckey],
                    lw=PLOT_LW[ckey],
                    label=label,
                )
            else:
                axc.plot(
                    xe[xmask],
                    y[xmask],
                    color=panel_colors[ckey],
                    lw=PLOT_LW[ckey],
                    label=label,
                )
        _plot_minimal_style(axc)
        axc.set_xlim(dist.plot_x_min, dist.plot_x_max)
        if row == 0:
            axc.set_title("CDF")
        axd.text(
            -0.34,
            0.5,
            dist.name,
            transform=axd.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontweight="bold",
        )
        if legend is None:
            legend = axc.get_legend_handles_labels()
    if legend:
        _, legend_labels = legend
        fig.legend(
            *legend,
            loc="lower left",
            bbox_to_anchor=(0.06, 0.015, 0.88, 0.035),
            ncol=max(1, len(legend_labels)),
            mode="expand",
            frameon=False,
            fontsize=12,
            handlelength=1.4,
            handletextpad=0.35,
            columnspacing=0.55,
            borderaxespad=0.0,
        )
    fig.subplots_adjust(
        left=0.08, right=0.99, top=0.93, bottom=0.115, wspace=0.3, hspace=0.42
    )
    _plot_save_fig(fig, path)
    plt.close(fig)


def _merge_parseval_histories(
    histories: list[dict[str, list[float]]],
) -> dict[str, list]:
    """Stack per-seed parseval series for CI shading (list-of-series per metric)."""
    if not histories:
        return {}
    out = {
        "step": list(histories[0]["step"]),
        "cramer_l2_dirac": [],
        "cf_l2_w_dirac": [],
    }
    for h in histories:
        out["cramer_l2_dirac"].append(list(h["cramer_l2_dirac"]))
        out["cf_l2_w_dirac"].append(list(h["cf_l2_w_dirac"]))
    return out
