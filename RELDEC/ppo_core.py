import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def collect_rollouts(envs, policy, critic, rollout_steps=512):
    # Sequential collection for simplicity
    trajectories = []
    
    # We assume envs are already reset
    # Track current state for each env
    states = [env._get_graph_state() for env in envs]
    
    steps_collected = 0
    while steps_collected < rollout_steps:
        for i, env in enumerate(envs):
            if steps_collected >= rollout_steps:
                break
                
            state = states[i]
            
            with torch.no_grad():
                action, log_prob = policy.get_action_and_log_prob(state)
                value = critic(state, env.current_iter, env.max_iter)
                
            next_state, reward, done, info = env.step(action)
            
            trajectories.append({
                'state': state,
                'action': action,
                'log_prob': log_prob,
                'reward': reward,
                'value': value,
                'done': done,
                'iter': env.current_iter - 1, # state was from before step
                'max_iter': env.max_iter
            })
            
            if done:
                # Randomize SNR on reset
                env.snr_db = np.random.uniform(0.5, 2.5)
                next_state = env.reset()
                
            states[i] = next_state
            steps_collected += 1
            
    return trajectories

class PPOTrainer:
    def __init__(self, env_fn, policy, critic, lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2, entropy_coef=0.01, 
                 value_loss_coef=0.5, max_grad_norm=0.5, ppo_epochs=4, batch_size=64, n_envs=8):
                 
        self.envs = [env_fn() for _ in range(n_envs)]
        for env in self.envs:
            env.reset()
            
        self.policy = policy
        self.critic = critic
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.optimizer_actor = optim.Adam(policy.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
        
    def train_iteration(self, rollout_steps=512):
        trajectories = collect_rollouts(self.envs, self.policy, self.critic, rollout_steps)
        
        # Compute GAE
        rewards = [t['reward'] for t in trajectories]
        values = [t['value'] for t in trajectories]
        dones = [t['done'] for t in trajectories]
        
        # Bootstrap value for last state
        with torch.no_grad():
            last_val = 0.0 # Approximation if episode ended exactly, else should query critic. We just use 0.0 for simplicity at boundary
            
        advantages = np.zeros(len(trajectories), dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(trajectories))):
            if t == len(trajectories) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_val
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t+1]
                
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + np.array(values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors/lists for PPO update
        states = [t['state'] for t in trajectories]
        actions = [t['action'] for t in trajectories]
        old_log_probs = torch.tensor([t['log_prob'] for t in trajectories], dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        iters = [t['iter'] for t in trajectories]
        max_iters = [t['max_iter'] for t in trajectories]
        
        dataset_size = len(trajectories)
        indices = np.arange(dataset_size)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        device = next(self.policy.parameters()).device
        old_log_probs = old_log_probs.to(device)
        returns = returns.to(device)
        advantages = advantages.to(device)
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                if start == end: continue
                batch_idx = indices[start:end]
                
                batch_states = [states[i] for i in batch_idx]
                batch_actions = [actions[i] for i in batch_idx]
                batch_iters = [iters[i] for i in batch_idx]
                batch_max_iters = [max_iters[i] for i in batch_idx]
                
                new_log_probs, entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                v_pred = self.critic.evaluate_values(batch_states, batch_iters, batch_max_iters)
                
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[batch_idx]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(v_pred, returns[batch_idx])
                entropy_loss = -entropy.mean()
                
                loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss
                
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                
        num_batches = self.ppo_epochs * (dataset_size // self.batch_size + 1)
        return {
            'actor_loss': total_actor_loss / num_batches,
            'critic_loss': total_critic_loss / num_batches,
            'entropy': total_entropy / num_batches,
            'avg_reward': np.mean(rewards)
        }
