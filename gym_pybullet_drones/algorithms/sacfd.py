from typing import Any, Dict, Tuple, Union
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy
from torch.nn import functional as F
from torch._C import device
import torch
import torch as th


class SACfD(SAC):
    def __init__(
            self,
            policy: str | type[SACPolicy],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1000000,
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: int | Tuple[int, str] = 1,
            gradient_steps: int = 1,
            action_noise: ActionNoise | None = None,
            replay_buffer_class: type[ReplayBuffer] | None = None,
            replay_buffer_kwargs: Dict[str, Any] | None = None,
            optimize_memory_usage: bool = False,
            ent_coef: str | float = "auto",
            target_update_interval: int = 1,
            target_entropy: str | float = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: str | None = None,
            policy_kwargs: Dict[str, Any] | None = None,
            verbose: int = 0,
            seed: int | None = None,
            device: device | str = "auto",
            _init_setup_model: bool = True):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model)

    # def train_actor(self, batch_size=64, num_steps=200_000):
    #     # Switch to train mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(True)
    #     # Update optimizers learning rate
    #     optimizers = [self.actor.optimizer, self.critic.optimizer]
    #     if self.ent_coef_optimizer is not None:
    #         optimizers += [self.ent_coef_optimizer]

    #     # Update learning rate according to lr schedule
    #     self._update_learning_rate(optimizers)

    #     ent_coef_losses, ent_coefs = [], []
    #     actor_losses, critic_losses = [], []

    #     for _ in range(num_steps):
    #         replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
    #         print(f"q:{replay_data.observations, replay_data.actions}")

    #         # We need to sample because `log_std` may have changed between two gradient steps
    #         if self.use_sde:
    #             self.actor.reset_noise()

    #         # Action by the current actor for the sampled state
    #         actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
    #         log_prob = log_prob.reshape(-1, 1)

    #         ent_coef_loss = None
    #         if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
    #             # Important: detach the variable from the graph
    #             # so we don't change it with other losses
    #             # see https://github.com/rail-berkeley/softlearning/issues/60
    #             ent_coef = th.exp(self.log_ent_coef.detach())
    #             ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
    #             ent_coef_losses.append(ent_coef_loss.item())
    #         else:
    #             ent_coef = self.ent_coef_tensor

    #         ent_coefs.append(ent_coef.item())

    #         # Optimize entropy coefficient, also called
    #         # entropy temperature or alpha in the paper
    #         if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
    #             self.ent_coef_optimizer.zero_grad()
    #             ent_coef_loss.backward()
    #             self.ent_coef_optimizer.step()

    #         with th.no_grad():
    #             # Select action according to policy
    #             next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
    #             # Compute the next Q values: min over all critics targets
    #             next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
    #             next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
    #             # add entropy term
    #             next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
    #             # td error + entropy term
    #             target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

    #         # Get current Q-values estimates for each critic network
    #         # using action from the replay buffer
    #         current_q_values = self.critic(replay_data.observations, replay_data.actions)
            
            
    #         # Compute critic loss
    #         critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
    #         assert isinstance(critic_loss, th.Tensor)  # for type checker
    #         critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

    #         # Optimize the critic
    #         self.critic.optimizer.zero_grad()
    #         critic_loss.backward()
    #         self.critic.optimizer.step()

    #         # Compute actor loss
    #         # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
    #         # Min over all critic networks
    #         # q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
    #         # min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
    #         # actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

    #         actor_loss = torch.nn.functional.mse_loss(actions_pi, replay_data.actions)
    #         print(f"q:{actions_pi, replay_data.actions}")

    #         actor_losses.append(actor_loss.item())

    #         # Optimize the actor
    #         self.actor.optimizer.zero_grad()
    #         actor_loss.backward()
    #         self.actor.optimizer.step()

    #         # Update target networks
    #         if True:
    #             polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
    #             # Copy running stats, see GH issue #996
    #             polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
