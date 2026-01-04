import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

device = "cuda" if torch.cuda.is_available() else "cpu"

class Dynamics(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers):
        super(Dynamics, self).__init__()

        self.model = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input):

        out, self.hidden_states = self.model(input.unsqueeze(0), self.hidden_states)
        return out
    
    def reset(self, dones=None):
        if dones is None:  # reset all hidden states
            self.hidden_states = None
        elif self.hidden_states is not None:  # reset hidden states of done environments
            if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                for hidden_state in self.hidden_states:
                    hidden_state[..., dones == 1, :] = 0.0
            else:
                self.hidden_states[..., dones == 1, :] = 0.0

    def detach_hidden_states(self, dones=None):
        if self.hidden_states is not None:
            if dones is None:  # detach all hidden states
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    self.hidden_states = tuple(hidden_state.detach() for hidden_state in self.hidden_states)
                else:
                    self.hidden_states = self.hidden_states.detach()
            else:  # detach hidden states of done environments
                if isinstance(self.hidden_states, tuple):  # tuple in case of LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_states[..., dones == 1, :] = self.hidden_states[..., dones == 1, :].detach()

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy