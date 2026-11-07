"""Categorical projection and action selection."""

from __future__ import annotations

import jax.numpy as jnp


def select_action_values(values: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
    if values.ndim == actions.ndim + 1:
        action_idx = jnp.expand_dims(actions, axis=-1)
        return jnp.take_along_axis(values, action_idx, axis=-1).squeeze(axis=-1)
    if values.ndim == actions.ndim + 2:
        action_idx = jnp.expand_dims(jnp.expand_dims(actions, axis=-1), axis=-1)
        action_idx = jnp.broadcast_to(action_idx, actions.shape + (1, values.shape[-1]))
        return jnp.take_along_axis(values, action_idx, axis=-2).squeeze(axis=-2)
    raise ValueError(
        f"Cannot select actions with values shape {values.shape} and actions shape {actions.shape}"
    )


def project_categorical(
    next_probs: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
    support: jnp.ndarray,
    gamma: float,
    v_min: float,
    v_max: float,
) -> jnp.ndarray:
    num_atoms = support.shape[0]
    flat_probs = next_probs.reshape(-1, num_atoms)
    flat_rewards = rewards.reshape(-1)
    flat_dones = dones.reshape(-1)
    delta_z = (v_max - v_min) / (num_atoms - 1)
    tz = flat_rewards[:, None] + gamma * (1.0 - flat_dones[:, None]) * support[None, :]
    tz = jnp.clip(tz, v_min, v_max)
    b = (tz - v_min) / delta_z
    lower = jnp.floor(b).astype(jnp.int32)
    upper = jnp.ceil(b).astype(jnp.int32)
    same = (upper == lower).astype(flat_probs.dtype)
    lower_w = flat_probs * (upper.astype(flat_probs.dtype) - b + same)
    upper_w = flat_probs * (b - lower.astype(flat_probs.dtype))
    batch_idx = jnp.arange(flat_probs.shape[0])[:, None]
    projected = jnp.zeros_like(flat_probs)
    projected = projected.at[batch_idx, lower].add(lower_w)
    projected = projected.at[batch_idx, upper].add(upper_w)
    return projected.reshape(next_probs.shape)
