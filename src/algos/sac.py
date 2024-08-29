import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
import random
import json
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci


class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T=10, json_file=None, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.T = self.env.scenario.thorizon
        self.s_acc, self.s_dem = self.get_scaling_factors()     # ADDED
        self.json_file = json_file
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs):
        x = torch.cat((
            torch.tensor([obs[0][n][self.env.time+1]*self.s_acc for n in self.env.region]).view(1, 1, self.env.nregions).float(),
            torch.tensor([[(obs[0][n][self.env.time+1] + self.env.dacc[n][t])*self.s_acc for n in self.env.region] \
                          for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregions).float(),
            torch.tensor([[sum([(self.env.scenario.demand_input[i,j][t])*(self.env.price[i,j][t])*self.s_dem \
                          for j in self.env.region]) for i in self.env.region] for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregions).float()),
              dim=1).squeeze(0).view(1+self.T+self.T, self.env.nregions).T

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
        # edge_index = torch.cat([torch.tensor([self.env.region]), torch.tensor([self.env.region])])    # Just local region information
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


class PairData(Data):
    """
    Store 2 graphs in one Data object (s_t and s_t+1)
    """

    def __init__(self, edge_index_s=None, x_s=None, reward=None, action=None, edge_index_t=None, x_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ReplayData:
    """
    Replay buffer for SAC agents
    """

    def __init__(self, device):
        self.device = device
        self.data_list = []
        self.rewards = []

    def store(self, data1, action, reward, data2):
        self.data_list.append(PairData(data1.edge_index, data1.x, torch.as_tensor(
            reward), torch.as_tensor(action), data2.edge_index, data2.x))
        self.rewards.append(reward)

    def size(self):
        return len(self.data_list)

    def sample_batch(self, batch_size=32, norm=False):
        data = random.sample(self.data_list, batch_size)
        if norm:
            mean = np.mean(self.rewards)
            std = np.std(self.rewards)
            batch = Batch.from_data_list(data, follow_batch=['x_s', 'x_t'])
            batch.reward = (batch.reward-mean)/(std + 1e-16)
            return batch.to(self.device)
        else:
            return Batch.from_data_list(data, follow_batch=['x_s', 'x_t']).to(self.device)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant

#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, deterministic=False):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1)
        if deterministic:
            action = (concentration) / (concentration.sum() + 1e-20)
            log_prob = None
        else:
            m = Dirichlet(concentration + 1e-20)
            action = m.rsample()
            log_prob = m.log_prob(action)
        return action, log_prob


class GNNActorLSTM(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lstm = nn.LSTM(in_channels, hidden_size, dropout=0.3)
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, deterministic=False):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x, _ = self.lstm(x)
        x = F.leaky_relu(self.lin1(x))
        x = F.softplus(self.lin2(x))
        concentration = x.squeeze(-1)
        if deterministic:
            action = (concentration) / (concentration.sum() + 1e-20)
            log_prob = None
        else:
            m = Dirichlet(concentration + 1e-20)
            action = m.rsample()
            log_prob = m.log_prob(action)
        return action, log_prob


#########################################
############## CRITIC ###################
#########################################


class GNNCritic1(nn.Module):
    """
    Architecture 1, GNN, Pointwise Multiplication, Readout, FC
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        action = action * 10
        action = action.unsqueeze(-1)  # (B,N,1)
        x = x * action  # pointwise multiplication (B,N,21)
        x = x.sum(dim=1)  # (B,21)
        x = F.relu(self.lin1(x))  # (B,H)
        x = F.relu(self.lin2(x))  # (B,H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic2(nn.Module):
    """
    Architecture 2, GNN, Readout, Concatenation, FC
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + act_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, 21)  # (B,N,21)
        x = torch.sum(x, dim=1)  # (B, 21)
        concat = torch.cat([x, action], dim=-1)  # (B, 21+N)
        x = F.relu(self.lin1(concat))  # (B,H)
        x = F.relu(self.lin2(x))  # (B,H)
        x = self.lin3(x).squeeze(-1)  # B
        return x


class GNNCritic3(nn.Module):
    """
    Architecture 3: Concatenation, GNN, Readout, FC
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(22, 22)
        self.lin1 = nn.Linear(22, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        cat = torch.cat([state, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        out = F.relu(self.conv1(cat, edge_index))
        x = out + cat
        x = x.reshape(-1, self.act_dim, 22)  # (B,N,22)
        x = F.relu(self.lin1(x))  # (B, H)
        x = F.relu(self.lin2(x))  # (B, H)
        x = torch.sum(x, dim=1)  # (B, 22)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic4(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + 1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic4LSTM(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lstm = nn.LSTM(in_channels + 1, hidden_size, dropout=0.3)
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        x, _ = self.lstm(concat)
        x = F.relu(self.lin1(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin2(x).squeeze(-1)  # (B)
        return x


class GNNCritic5(nn.Module):
    """
    Architecture 5, GNN, Pointwise Multiplication, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        action = action + 1
        action = action.unsqueeze(-1)  # (B,N,1)
        x = x * action  # pointwise multiplication (B,N,21)
        x = F.relu(self.lin1(x))  # (B,N,H)
        x = F.relu(self.lin2(x))  # (B,N,H)
        x = x.sum(dim=1)  # (B,H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


#########################################
############## A2C AGENT ################
#########################################


class SAC(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        env,
        input_size,
        hidden_size=32,
        alpha=0.2,
        gamma=0.99,
        polyak=0.995,
        batch_size=128,
        p_lr=3e-4,
        q_lr=1e-3,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=-1,
        min_q_weight=1,
        deterministic_backup=False,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
        min_q_version=3,
        clip=200,
        critic_version=4,
    ):
        super(SAC, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.path = None
        self.act_dim = env.nregions

        # SAC parameters
        self.alpha = alpha
        self.polyak = polyak
        self.env = env
        self.BATCH_SIZE = batch_size
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.gamma = gamma
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.min_q_version = min_q_version
        self.clip = clip

        # conservative Q learning parameters
        self.num_random = 10
        self.temp = 1.0
        self.min_q_weight = min_q_weight
        if lagrange_thresh == -1:
            self.with_lagrange = False
        else:
            print("using lagrange")
            self.with_lagrange = True
        self.deterministic_backup = deterministic_backup
        self.step = 0
        self.nodes = env.nregions

        self.replay_buffer = ReplayData(device=device)
        # nnets
        self.actor = GNNActor(self.input_size, self.hidden_size, act_dim=self.act_dim)
    
        if critic_version == 1:
            GNNCritic = GNNCritic1
        if critic_version == 2:
            GNNCritic = GNNCritic2
        if critic_version == 3:
            GNNCritic = GNNCritic3
        if critic_version == 4:
            GNNCritic = GNNCritic4
        if critic_version == 5:
            GNNCritic = GNNCritic5
        if critic_version == 6:
            GNNCritic = GNNCritic4LSTM
            self.actor = GNNActorLSTM(self.input_size, self.hidden_size, act_dim=self.act_dim)

        self.critic1 = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic2 = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim)
        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh  # lagrange treshhold
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = torch.optim.Adam(
                self.log_alpha_prime.parameters(),
                lr=self.p_lr,
            )

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(self.act_dim).item()
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=1e-3
            )

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, data, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(data.x, data.edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy()[0]
        return list(a)

    def compute_loss_q(self, data):
        (
            state_batch,
            edge_index,
            next_state_batch,
            edge_index2,
            reward_batch,
            action_batch,
        ) = (
            data.x_s,
            data.edge_index_s,
            data.x_t,
            data.edge_index_t,
            data.reward,
            data.action.reshape(-1, self.nodes),
        )

        q1 = self.critic1(state_batch, edge_index, action_batch)
        q2 = self.critic2(state_batch, edge_index, action_batch)
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor(next_state_batch, edge_index2)
            q1_pi_targ = self.critic1_target(next_state_batch, edge_index2, a2)
            q2_pi_targ = self.critic2_target(next_state_batch, edge_index2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            backup = reward_batch + self.gamma * (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        return loss_q1, loss_q2

    def compute_loss_pi(self, data):
        state_batch, edge_index = (
            data.x_s,
            data.edge_index_s,
        )

        actions, logp_a = self.actor(state_batch, edge_index)
        q1_1 = self.critic1(state_batch, edge_index, actions)
        q2_a = self.critic2(state_batch, edge_index, actions)
        q_a = torch.min(q1_1, q2_a)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (logp_a + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()

        loss_pi = (self.alpha * logp_a - q_a).mean()
        return loss_pi

    def update(self, data):
        loss_q1, loss_q2 = self.compute_loss_q(data)

        self.optimizers["c1_optimizer"].zero_grad()

        nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        loss_q1.backward()
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.optimizers["c2_optimizer"].step()

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        # one gradient descent step for policy network
        self.optimizers["a_optimizer"].zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.optimizers["a_optimizer"].step()

        # Unfreeze Q-networks
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(critic1_params, lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(critic2_params, lr=self.q_lr)

        return optimizers

    def test_agent(self, test_episodes, env, cplexpath, matching_steps, agent_name, parser):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        episode_rebalanced_vehicles = []
        for _ in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            eps_rebalancing_veh = 0
            actions = []
            done = False
            # Reset the environment
            obs = env.reset()   # initialize environment
            if self.env.scenario.is_meso:
                try:
                    traci.simulationStep()
                except Exception as e:
                    print(f"FatalTraCIError during initial step: {e}")
                    traci.close()
                    break
            while not done:
                sumo_step = 0
                try:
                    while sumo_step < matching_steps:
                        traci.simulationStep()
                        sumo_step += 1
                except Exception as e:
                    print(f"FatalTraCIError during matching steps: {e}")
                    traci.close()
                    break
                obs, paxreward, done, info = env.pax_step(CPLEXPATH=cplexpath, PATH=f'scenario_lux/{agent_name}')
                eps_reward += paxreward

                o = parser.parse_obs(obs)

                action_rl = self.select_action(o, deterministic=True)
                actions.append(action_rl)

                desired_acc = {env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + env.tstep)) for i in range(len(env.region))}

                reb_action = solveRebFlow(env, f'scenario_lux/{agent_name}', desired_acc, cplexpath)

                # Take action in environment
                try:
                    _, rebreward, done, info = env.reb_step(reb_action)
                except Exception as e:
                    print(f"FatalTraCIError during rebalancing step: {e}")
                    traci.close()
                    break

                eps_reward += rebreward
                eps_served_demand += info["served_demand"]
                eps_rebalancing_cost += info["rebalancing_cost"]
                eps_rebalancing_veh += info["rebalanced_vehicles"]
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)
            episode_rebalanced_vehicles.append(eps_rebalancing_veh)

            # stop episode if terminating conditions are met
            if done:
                traci.close()

        return (
            np.mean(episode_reward),
            np.mean(episode_served_demand),
            np.mean(episode_rebalancing_cost),
        )

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["model"].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
