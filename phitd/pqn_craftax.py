"""Recurrent PQN baseline on Craftax. Adapted from PureJaxQL (Apache-2.0)."""

from functools import partial
from typing import Any

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name
from flax.training.train_state import TrainState

from phitd.norm import BatchRenorm
from phitd.wrappers import BatchEnvWrapper, LogWrapper, OptimisticResetVecEnvWrapper


class ScannedRNN(nn.Module):
    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = rnn_state[0].shape[-1]
        init_rnn_state = self.initialize_carry(hidden_size, *resets.shape)
        rnn_state = jax.tree_util.tree_map(
            lambda init, old: jnp.where(resets[:, np.newaxis], init, old),
            init_rnn_state,
            rnn_state,
        )
        new_rnn_state, y = nn.OptimizedLSTMCell(hidden_size)(rnn_state, ins)
        return (new_rnn_state, y)

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    norm_input: bool = False
    norm_type: str = "layer_norm"
    dueling: bool = False
    add_last_action: bool = False

    @nn.compact
    def __call__(self, hidden, x, done, last_action, train: bool = False):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x
        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            _ = BatchRenorm(use_running_average=not train)(x)
        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)
        if self.add_last_action:
            last_action = jax.nn.one_hot(last_action, self.action_dim)
            x = jnp.concatenate([x, last_action], axis=-1)
        new_hidden = []
        for i in range(self.num_rnn_layers):
            rnn_in = (x, done)
            hidden_aux, x = ScannedRNN()(hidden[i], rnn_in)
            new_hidden.append(hidden_aux)
        q_vals = nn.Dense(self.action_dim)(x)
        return (new_hidden, q_vals)

    def initialize_carry(self, *batch_size):
        return [
            ScannedRNN.initialize_carry(self.hidden_size, *batch_size)
            for _ in range(self.num_rnn_layers)
        ]


@chex.dataclass(frozen=True)
class Transition:
    last_hs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    last_done: chex.Array
    last_action: chex.Array
    q_vals: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    assert config["NUM_STEPS"] * config["NUM_ENVS"] % config["NUM_MINIBATCHES"] == 0, (
        "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"
    )
    basic_env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = basic_env.default_params
    log_env = LogWrapper(basic_env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
        test_env = OptimisticResetVecEnvWrapper(
            log_env,
            num_envs=config["TEST_NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["TEST_NUM_ENVS"]),
        )
    else:
        env = BatchEnvWrapper(log_env, num_envs=config["NUM_ENVS"])
        test_env = BatchEnvWrapper(log_env, num_envs=config["TEST_NUM_ENVS"])

    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),
            greedy_actions,
        )
        return chosen_actions

    def train(rng, seed_idx):
        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            config["EPS_DECAY"] * config["NUM_UPDATES_DECAY"],
        )
        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=config["NUM_UPDATES_DECAY"]
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        network = RNNQNetwork(
            action_dim=env.action_space(env_params).n,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            num_rnn_layers=config.get("NUM_RNN_LAYERS", 1),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            add_last_action=config.get("ADD_LAST_ACTION", False),
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros((1, 1, *env.observation_space(env_params).shape)),
                jnp.zeros((1, 1)),
                jnp.zeros((1, 1)),
            )
            init_hs = network.initialize_carry(1)
            network_variables = network.init(rng, init_hs, *init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )
            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        def _update_step(runner_state, unused):
            train_state, memory_transitions, expl_state, test_metrics, rng = (
                runner_state
            )

            def _step_env(carry, _):
                hs, last_obs, last_done, last_action, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = last_obs[np.newaxis]
                _done = last_done[np.newaxis]
                _last_action = last_action[np.newaxis]
                new_hs, q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                )
                q_vals = q_vals.squeeze(axis=0)
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )
                transition = Transition(
                    last_hs=hs,
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    last_done=last_done,
                    last_action=last_action,
                    q_vals=q_vals,
                )
                return (
                    (new_hs, new_obs, new_done, new_action, new_env_state, rng),
                    (transition, info),
                )

            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env, (*expl_state, _rng), None, config["NUM_STEPS"]
            )
            expl_state = tuple(expl_state)
            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )
            memory_transitions = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x[config["NUM_STEPS"] :], y], axis=0),
                memory_transitions,
                transitions,
            )

            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch):
                    train_state, rng = carry
                    hs = jax.tree_util.tree_map(lambda x: x[0], minibatch.last_hs)
                    agent_in = (
                        minibatch.obs,
                        minibatch.last_done,
                        minibatch.last_action,
                    )

                    def _compute_targets(last_q, q_vals, reward, done):

                        def _get_target(lambda_returns_and_next_q, rew_q_done):
                            reward, q, done = rew_q_done
                            lambda_returns, next_q = lambda_returns_and_next_q
                            target_bootstrap = (
                                reward + config["GAMMA"] * (1 - done) * next_q
                            )
                            delta = lambda_returns - next_q
                            lambda_returns = (
                                target_bootstrap
                                + config["GAMMA"] * config["LAMBDA"] * delta
                            )
                            lambda_returns = (1 - done) * lambda_returns + done * reward
                            next_q = jnp.max(q, axis=-1)
                            return ((lambda_returns, next_q), lambda_returns)

                        lambda_returns = (
                            reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                        )
                        last_q = jnp.max(q_vals[-1], axis=-1)
                        _, targets = jax.lax.scan(
                            _get_target,
                            (lambda_returns, last_q),
                            jax.tree_util.tree_map(
                                lambda x: x[:-1], (reward, q_vals, done)
                            ),
                            reverse=True,
                        )
                        targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                        return targets

                    def _loss_fn(params):
                        (_, q_vals), updates = partial(
                            network.apply, train=True, mutable=["batch_stats"]
                        )(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            hs,
                            *agent_in,
                        )
                        target_q_vals = jax.lax.stop_gradient(q_vals)
                        last_q = target_q_vals[-1].max(axis=-1)
                        target = _compute_targets(
                            last_q,
                            target_q_vals[:-1],
                            minibatch.reward[:-1],
                            minibatch.done[:-1],
                        ).reshape(-1)
                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals, jnp.expand_dims(minibatch.action, axis=-1), axis=-1
                        ).squeeze(axis=-1)
                        chosen_action_qvals = chosen_action_qvals[:-1].reshape(-1)
                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()
                        return (loss, (updates, chosen_action_qvals))

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    grad_norm = optax.global_norm(grads)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return ((train_state, rng), (loss, qvals, grad_norm))

                def preprocess_transition(x, rng):
                    x = jax.random.permutation(rng, x, axis=1)
                    x = x.reshape(
                        x.shape[0], config["NUM_MINIBATCHES"], -1, *x.shape[2:]
                    )
                    x = jnp.swapaxes(x, 0, 1)
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), memory_transitions
                )
                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals, grad_norm) = jax.lax.scan(
                    _learn_phase, (train_state, rng), minibatches
                )
                return ((train_state, rng), (loss, qvals, grad_norm))

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals, grad_norm) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
                "grad_norm": grad_norm.mean(),
            }
            done_infos = jax.tree_util.tree_map(
                lambda x: (
                    (x * infos["returned_episode"]).sum()
                    / infos["returned_episode"].sum()
                ),
                infos,
            )
            metrics.update(done_infos)
            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})
            if not config.get("LOG_ACHIEVEMENTS", False):
                metrics = {
                    k: v for k, v in metrics.items() if "achievement" not in k.lower()
                }
            runner_state = (
                train_state,
                memory_transitions,
                tuple(expl_state),
                test_metrics,
                rng,
            )
            return (runner_state, metrics)

        def get_test_metrics(train_state, rng):
            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _greedy_env_step(step_state, _):
                hs, last_obs, last_done, last_action, env_state, rng = step_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = last_obs[np.newaxis]
                _done = last_done[np.newaxis]
                _last_action = last_action[np.newaxis]
                new_hs, q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                )
                q_vals = q_vals.squeeze(axis=0)
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    _rng, env_state, new_action, env_params
                )
                step_state = (new_hs, new_obs, new_done, new_action, new_env_state, rng)
                return (step_state, info)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.reset(_rng, env_params)
            init_done = jnp.zeros(config["TEST_NUM_ENVS"], dtype=bool)
            init_action = jnp.zeros(config["TEST_NUM_ENVS"], dtype=int)
            init_hs = network.initialize_carry(config["TEST_NUM_ENVS"])
            step_state = (init_hs, init_obs, init_done, init_action, env_state, _rng)
            step_state, infos = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            done_infos = jax.tree_util.tree_map(
                lambda x: (
                    (x * infos["returned_episode"]).sum()
                    / infos["returned_episode"].sum()
                ),
                infos,
            )
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        init_dones = jnp.zeros(config["NUM_ENVS"], dtype=bool)
        init_action = jnp.zeros(config["NUM_ENVS"], dtype=int)
        init_hs = network.initialize_carry(config["NUM_ENVS"])
        expl_state = (init_hs, obs, init_dones, init_action, env_state)

        def _random_step(carry, _):
            hs, last_obs, last_done, last_action, env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _obs = last_obs[np.newaxis]
            _done = last_done[np.newaxis]
            _last_action = last_action[np.newaxis]
            new_hs, q_vals = network.apply(
                {"params": train_state.params, "batch_stats": train_state.batch_stats},
                hs,
                _obs,
                _done,
                _last_action,
                train=False,
            )
            q_vals = q_vals.squeeze(axis=0)
            _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
            eps = jnp.full(config["NUM_ENVS"], 1.0)
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
            new_obs, new_env_state, reward, new_done, info = env.step(
                rng_s, env_state, new_action, env_params
            )
            transition = Transition(
                last_hs=hs,
                obs=last_obs,
                action=new_action,
                reward=config.get("REW_SCALE", 1) * reward,
                done=new_done,
                last_done=last_done,
                last_action=last_action,
                q_vals=q_vals,
            )
            return (
                (new_hs, new_obs, new_done, new_action, new_env_state, rng),
                transition,
            )

        rng, _rng = jax.random.split(rng)
        (*expl_state, rng), memory_transitions = jax.lax.scan(
            _random_step,
            (*expl_state, _rng),
            None,
            config["MEMORY_WINDOW"] + config["NUM_STEPS"],
        )
        expl_state = tuple(expl_state)
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, memory_transitions, expl_state, test_metrics, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train
