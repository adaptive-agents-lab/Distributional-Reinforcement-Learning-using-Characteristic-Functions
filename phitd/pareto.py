"""Truncated Pareto frequency sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sample_truncated_pareto_alpha_1(
    key: jax.Array,
    num_samples: int,
    omega_max: float | jnp.ndarray,
    omega_min: float | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Sample ω from truncated Pareto (α=1) on ``[omega_min, omega_max]``.

    Args:
        key: PRNG key.
        num_samples: number of frequencies.
        omega_max: upper truncation ``B``.
        omega_min: lower bound ``x_m`` (scale). If None, uses ``max(1e-8, omega_max * 1e-6)``.

    Returns:
        (num_samples,) positive frequencies.
    """
    B = jnp.asarray(omega_max, dtype=jnp.float32)
    if omega_min is None:
        xm = jnp.maximum(jnp.float32(1e-08), B * jnp.float32(1e-06))
    else:
        xm = jnp.asarray(omega_min, dtype=jnp.float32)
    xm = jnp.maximum(xm, jnp.float32(1e-12))
    xm = jnp.minimum(xm, B * jnp.float32(1.0 - 1e-06))
    u = jax.random.uniform(key, shape=(num_samples,), dtype=jnp.float32)
    denom = 1.0 - u * (1.0 - xm / B)
    return xm / denom
