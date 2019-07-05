import numpy as np
from agents.actor import Actor
from agents.critic import Critic
from agents.ou_noise import OUNoise
from agents.replay_buffer import ReplayBuffer


class RLA():
    """ Reinfocement learning agent"""
    
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        #actor model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        
        #Critic model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        #Initialize target model params with local params
        self.critic_target.model.set_weights(
                self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(
                self.actor_local.model.get_weights())
        
        #Initialize noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        #Replay memory Initialization
        self.buffer_size, self.batch_size = 2000000, 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        #Initialize algorithm parameters
        self.gamma, self.tau = 0.95, 0.001
        
        #Initialize scores
        self.score, self.best_score = -np.inf, -np.inf
    
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.score = 0
        return state
    
    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state, action, reward, next_state, done)
        
        #Learn from samples in memory if they are greater than batch size
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
        
        #Preserve state as last_state
        self.last_state = next_state
        
        #Update score with reward from this step
        self.score += reward
        
        if done:
            #Preserve best score
            if self.score > self.best_score:
                self.best_score = self.score
        
    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())
    
    def learn(self, experiences):
        #Convert experiences seperate arrays
        states = np.vstack([exp.state for exp in experiences if exp is not None])
        actions = np.array([exp.action for exp in experiences if exp is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([exp.reward for exp in experiences if exp is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([exp.done for exp in experiences if exp is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([exp.next_state for exp in experiences if exp is not None])
        
        #predict next_state actions and Q values from target model...
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states,actions], y=Q_targets)
        
        #Train local actor model
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])
        
        #Update target models
        self.update(self.critic_local.model, self.critic_target.model)
        self.update(self.actor_local.model, self.actor_target.model)
        
    def update(self, local_model, target_model):
        """Update model parameters"""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
        
        
        
        
        
        
        
        
        
        