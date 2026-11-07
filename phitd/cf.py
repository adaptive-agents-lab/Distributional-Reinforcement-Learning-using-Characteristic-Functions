"""Characteristic functions, return moments, and frequency samplers."""

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from phitd.pareto import sample_truncated_pareto_alpha_1


def build_categorical_cf(probs, support, omegas):
    """Characteristic function of a categorical / discrete distribution on fixed atoms.

    φ(ω) = Σ_a p_a · exp(i · ω · z_a) with atoms ``support[a] = z_a``.

    Args:
        probs:   (..., num_atoms) — typically softmax probabilities per row.
        support: (num_atoms,) — atom locations (same convention as C51 ``support``).
        omegas:  (num_omega,) — real frequencies (≥ 0).

    Returns:
        Complex array (..., num_omega).
    """
    phase = omegas[:, None] * support[None, :]
    kernels = jnp.exp(1j * phase)
    return jnp.einsum("...a,na->...n", probs, kernels, optimize=True)


def build_dirac_mixture_cf(pi, mu, omegas):
    """Characteristic function of a mixture of Dirac masses (learned atom locations).

    φ(ω) = Σ_k π_k · exp(i · ω · μ_k). Same mixture structure as MoG but without Gaussian
    smoothing (used by Phi-TD / ``F_m`` on MinAtar).
    """
    pi_e = pi[..., None, :, :]
    mu_e = mu[..., None, :, :]
    w = omegas[..., None, None]
    phase = w * mu_e
    comp_cf = jnp.exp(1j * phase)
    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def build_quantile_cf(quantiles, omegas):
    """Characteristic function of the empirical / quantile-histogram distribution.

    Mass 1/M on each ordered quantile location θ_j (``F_{Q,m}`` convention):

        φ(ω) = (1/M) Σ_j exp(i · ω · θ_j).

    Args:
        quantiles: (..., M) learned quantile values.
        omegas: (N,) real frequencies.

    Returns:
        Complex array (..., N).
    """

    def for_omega(w: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.exp(1j * w * quantiles), axis=-1)

    stacked = jax.vmap(for_omega)(omegas)
    return jnp.moveaxis(stacked, 0, -1)


def build_mog_cf(pi, mu, sigma, omegas):
    """Construct the Mixture-of-Gaussians characteristic function.

    φ(ω) = Σ_k π_k · exp(i·ω·μ_k − ½·ω²·σ²_k)

    This CF is **always valid** by construction:
      - φ(0) = Σ π_k = 1  (since π comes from softmax)
      - |φ(ω)| ≤ Σ π_k |exp(...)| ≤ Σ π_k = 1
    And the Bellman operator maps MoG → MoG (closure).

    Args:
        pi:     (..., action_dim, M)  — mixture weights (sum to 1 over M).
        mu:     (..., action_dim, M)  — component means.
        sigma:  (..., action_dim, M)  — component std devs (positive).
        omegas: (N,) — sampled frequencies (positive, exploiting conjugate symmetry).

    Returns:
        Complex array (..., N, action_dim) — CF values at each frequency.
    """
    pi_e = pi[..., None, :, :]
    mu_e = mu[..., None, :, :]
    sigma_e = sigma[..., None, :, :]
    w = omegas[..., None, None]
    phase = w * mu_e
    decay = -0.5 * w**2 * sigma_e**2
    comp_cf = jnp.exp(decay + 1j * phase)
    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def mog_q_values(pi, mu):
    """Closed-form Q from MoG parameters: Q(s, a) = E[G] = Σ_k π_k · μ_k."""
    return jnp.sum(pi * mu, axis=-1)


def build_cauchy_mixture_cf(pi, loc, scale, omegas):
    """Characteristic function of a mixture of Cauchy(loc_k, scale_k).

    φ(ω) = Σ_k π_k · exp(i·ω·loc_k − scale_k·|ω|).

    Under affine returns Z = r + γ X with X Cauchy(loc, scale), Z is Cauchy(r + γ·loc, γ·scale),
    matching the same Bellman closure pattern as MoG (scale scales with γ).

    Args:
        pi:    (..., action_dim, M) — mixture weights.
        loc:   (..., action_dim, M) — Cauchy locations.
        scale: (..., action_dim, M) — positive scale parameters.
        omegas: (N,) — real frequencies.

    Returns:
        Complex array (..., N, action_dim).
    """
    pi_e = pi[..., None, :, :]
    loc_e = loc[..., None, :, :]
    scale_e = scale[..., None, :, :]
    w = omegas[..., None, None]
    phase = w * loc_e
    decay = -scale_e * jnp.abs(w)
    comp_cf = jnp.exp(decay + 1j * phase)
    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def build_gamma_mixture_cf(pi, shape, scale, omegas):
    """Characteristic function of a mixture of Gamma(shape_k, scale_k).

    For X ~ Gamma(α, θ) (shape α, scale θ): φ_X(ω) = (1 − i·θ·ω)^(−α).

    φ_mixture(ω) = Σ_k π_k · (1 − i·θ_k·ω)^(−α_k).

    For returns Z = r + γ X, φ_Z(ω) = exp(i·r·ω) · φ_X(γω); equivalently per-component
    (α, θ) ↦ (α, γ·θ) with an outer exp(i·r·ω) factor (handled in the trainer).

    Args:
        pi:     (..., action_dim, M) — mixture weights.
        shape:  (..., action_dim, M) — Gamma shape α_k > 0.
        scale:  (..., action_dim, M) — Gamma scale θ_k > 0.
        omegas: (N,) — real frequencies.

    Returns:
        Complex array (..., N, action_dim).
    """
    pi_e = pi[..., None, :, :]
    shape_e = shape[..., None, :, :]
    scale_e = scale[..., None, :, :]
    w = omegas[..., None, None]
    z = (1.0 - 1j * scale_e * w).astype(jnp.complex64)
    comp_cf = jnp.power(z, -shape_e)
    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def gamma_mixture_q_values(pi, shape, scale):
    """Expectation Q(s, a) = E[G] = Σ_k π_k · α_k · θ_k for Gamma components."""
    return jnp.sum(pi * shape * scale, axis=-1)


def build_laplace_mixture_cf(pi, loc, scale, omegas):
    """Characteristic function of a mixture of Laplace(location μ, scale b).

    φ(ω) = Σ_k π_k · exp(i·μ_k·ω) / (1 + b_k² ω²).

    For Z = r + γ X with Laplace(μ, b), Z ~ Laplace(r + γ μ, γ b), same affine closure as MoG.

    Args:
        pi:    (..., action_dim, M) — mixture weights.
        loc:   (..., action_dim, M) — locations (means).
        scale: (..., action_dim, M) — scale b_k > 0.
        omegas: (N,) — real frequencies.

    Returns:
        Complex array (..., N, action_dim).
    """
    pi_e = pi[..., None, :, :]
    loc_e = loc[..., None, :, :]
    scale_e = scale[..., None, :, :]
    w = omegas[..., None, None]
    denom = 1.0 + (scale_e * w) ** 2
    comp_cf = jnp.exp(1j * loc_e * w) / denom
    phi = jnp.sum(pi_e * comp_cf, axis=-1)
    return phi


def sample_frequencies(
    key,
    num_samples,
    omega_max,
    scale=None,
    distribution="half_laplacian",
    gaussian_mean=0.0,
    gaussian_std=1.0,
    omega_min=None,
):
    """Sample positive ω frequencies for CF losses.

    Supported distributions:
      - ``half_laplacian`` / ``half_laplace``: truncated exponential on ``[0, omega_max]``.
      - ``pareto_1`` / ``pareto_alpha_1``: truncated Pareto with shape α=1 on
        ``[omega_min, omega_max]``. Pass ``omega_min`` explicitly (defaults inside
        ``sample_truncated_pareto_alpha_1``). Intended for unweighted CF MSE (Cramér-type objective).
      - ``uniform``: uniform on ``[0, omega_max]``.
      - ``half_gaussian``: absolute Gaussian ``|N(mean, std)|``, clipped to ``omega_max``.

    Notes:
      - Frequencies are always non-negative (conjugate symmetry in CF space).
    """
    distribution = str(distribution).lower()
    if distribution == "half_laplace":
        distribution = "half_laplacian"
    if distribution == "half_laplacian":
        if scale is None:
            scale = omega_max / 3.0
        u = jax.random.uniform(key, shape=(num_samples,))
        max_cdf = 1.0 - jnp.exp(-omega_max / scale)
        return -scale * jnp.log(1.0 - u * max_cdf)
    if distribution == "uniform":
        return jax.random.uniform(
            key, shape=(num_samples,), minval=0.0, maxval=omega_max
        )
    if distribution == "half_gaussian":
        if gaussian_mean != 0.0:
            raise ValueError("half_gaussian currently supports gaussian_mean=0.0 only.")
        if gaussian_std <= 0.0:
            raise ValueError("gaussian_std must be > 0 for half_gaussian.")
        u = jax.random.uniform(key, shape=(num_samples,))
        cdf_max = jsp.erf(omega_max / (gaussian_std * jnp.sqrt(2.0)))
        cdf_max = jnp.clip(cdf_max, 1e-08, 1.0 - 1e-08)
        omega = gaussian_std * jnp.sqrt(2.0) * jsp.erfinv(u * cdf_max)
        return jnp.clip(omega, 0.0, omega_max)
    if distribution in ("pareto_1", "pareto_alpha_1"):
        return sample_truncated_pareto_alpha_1(
            key, num_samples, omega_max, omega_min=omega_min
        )
    raise ValueError(
        f"Unknown distribution for sample_frequencies: {distribution}. Supported: half_laplacian (alias half_laplace), pareto_1, uniform, half_gaussian."
    )
