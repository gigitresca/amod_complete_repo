"""
A2C-GNN
-------
This file contains the A2C-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import grid
from collections import namedtuple
from src.misc.utils import dictsum

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.97
args.log_interval = 10

#########################################
############## PARSER ###################
#########################################

class GNNParser():
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """
    def __init__(self, env, grid_h=4, grid_w=4, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.T = self.env.scenario.thorizon
        self.s_acc, self.s_dem = self.get_scaling_factors()     # ADDED
        self.grid_h = grid_h
        self.grid_w = grid_w
        
    def parse_obs(self, obs):
        x = torch.cat((
            torch.tensor([obs[0][n][self.env.time+1]*self.s_acc for n in self.env.region]).view(1, 1, self.env.nregions).float(),
            torch.tensor([[(obs[0][n][self.env.time+1] + self.env.dacc[n][t])*self.s_acc for n in self.env.region] \
                          for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregions).float(),
            torch.tensor([[sum([(self.env.scenario.demand_input[i,j][t])*(self.env.price[i,j][t])*self.s_dem \
                          for j in self.env.region]) for i in self.env.region] for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregions).float()),
              dim=1).squeeze(0).view(21, self.env.nregions).T

        ###################
        # ADDED
        # Edge index self-connected tensor definition
        origin = []
        destination = []
        for o in range(self.env.scenario.adjacency_matrix.shape[0]):
            for d in range(self.env.scenario.adjacency_matrix.shape[1]):
                if self.env.scenario.adjacency_matrix[o, d] == 1:
                    origin.append(o)
                    destination.append(d)

        edge_index = torch.cat([torch.tensor([origin]), torch.tensor([destination])])
        # edge_index = torch.cat([torch.tensor([self.env.region]), torch.tensor([self.env.region])])        # just local region information
        ##################


        data = Data(x, edge_index)
        return data

    def get_scaling_factors(self):
        t0 = 0
        tf = self.env.scenario.duration
        time = [t for t in range(t0, tf)]
        acc_tot = (self.env.acc[0][0] * self.env.nregions)
        demand = self.env.scenario.demand_input
        price = self.env.scenario.price
        demand_max = max([max([demand[key][t] for key in demand]) for t in time])
        price_max = max([max([price[key][t] for key in price]) for t in time])
        return 2/acc_tot, 1/(1.2 * demand_max * price_max)

    
#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)
    
    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

#########################################
############## CRITIC ###################
#########################################

class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)
    
    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x 
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(self, env, input_size, eps=np.finfo(np.float32).eps.item(), device=torch.device("cpu")):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device
        
        self.actor = GNNActor(self.input_size, self.hidden_size)
        self.critic = GNNCritic(self.input_size, self.hidden_size)
        self.obs_parser = GNNParser(self.env)
        
        self.optimizers = self.configure_optimizers()
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)
        
    def forward(self, obs, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x = self.parse_obs(obs).to(self.device)
        
        # actor: computes concentration parameters of a Dirichlet distribution
        a_out = self.actor(x)
        concentration = F.softplus(a_out).reshape(-1) + jitter

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, value
    
    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state
    
    def select_action(self, obs):
        concentration, value = self.forward(obs)
        
        m = Dirichlet(concentration)
        
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), value))
        return list(action.cpu().numpy())

    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss 
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

        return [loss.item() for loss in value_losses], [loss.item() for loss in policy_losses]
    
    def learn(self, cfg):
        log = {'train_reward': [], 
           'train_served_demand': [], 
           'train_reb_cost': [],
           'train_reb_vehicles': [],
           'train_policy_losses': [],
           'train_value_losses': []}

        train_episodes = cfg.max_episodes #set max number of training episodes
        T = cfg.max_steps #set episode length
        epochs = trange(train_episodes)     # epoch iterator
        best_reward = -np.inf   # set best reward
        self.train()   # set model in train mode
        
        for i_episode in epochs:
    
            # Initialize the reward
            episode_reward = 0
            episode_served_demand = 0
            episode_rebalancing_cost = 0
            episode_rebalanced_vehicles = 0
            # Reset the environment
            obs = self.env.reset()  # initialize environment
            for step in range(T):
                # take matching step (Step 1 in paper)
                action_rl = self.select_action(obs)
                desired_acc = {self.env.region[i]: int(action_rl[i] * dictsum(self.env.acc, self.env.time+self.env.tstep))for i in range(len(self.env.region))}
                # solve minimum rebalancing distance problem (Step 3 in paper)
                reb_action = self.solveRebFlow(self.env, f'sac/scenario_lux/{self.cfg.agent_name}', desired_acc, self.cfg.cplexpath)

                obs, rew, done, info = self.env.step(reb_action)

                self.rewards.append(rew)
                # track performance over episode
                episode_served_demand += info['served_demand']
                episode_rebalancing_cost += info['rebalancing_cost']
                episode_rebalanced_vehicles += info['rebalanced_vehicles']
                # stop episode if terminating conditions are met
                if done:
                    break
            
            # Training step only if the episode was completed
            if step < (args.duration * 60) / args.matching_tstep - 1:
                epochs.set_description(f"Episode {i_episode + 1} | Not completed")
                continue
            # perform on-policy backprop
            p_loss, v_loss = self.training_step()
            log['train_policy_losses'].extend(p_loss)
            log['train_value_losses'].extend(v_loss)

            # Send current statistics to screen
            epochs.set_description(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}  | Reb. Veh: {episode_rebalanced_vehicles:.2f}")
            # Checkpoint best performing model
            if episode_reward >= best_reward:
                self.save_checkpoint(path=f"./{args.directory}/ckpt/scenario_lux/{args.agent_name}.pth")
                best_reward = episode_reward
            # Log KPIs
            log['train_reward'].append(episode_reward)
            log['train_served_demand'].append(episode_served_demand)
            log['train_reb_cost'].append(episode_rebalancing_cost)
            log['train_reb_vehicles'].append(episode_rebalanced_vehicles)
            self.log(log, path=f"./{args.directory}/rl_logs/scenario_lux/{args.agent_name}.pth")

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers['a_optimizer'] = torch.optim.Adam(actor_params, lr=1e-4)
        optimizers['c_optimizer'] = torch.optim.Adam(critic_params, lr=1e-4)
        return optimizers
    
    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path='ckpt.pth'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])
    
    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)
