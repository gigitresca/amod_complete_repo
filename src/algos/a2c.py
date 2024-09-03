import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from src.misc.utils import dictsum
from src.algos.gnn_critic import GNNValue
from src.algos.gnn_actor import GNNActor
from src.algos.reb_flow_solver import solveRebFlow
import os 
from tqdm import trange
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
args = namedtuple('args', ('render', 'gamma', 'log_interval'))
args.render= True
args.gamma = 0.97
args.log_interval = 10
#########################################
############## A2C AGENT ################
#########################################

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(self, env, input_size, parser, eps=np.finfo(np.float32).eps.item(), device=torch.device("cpu")):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device
        
        self.actor = GNNActor(self.input_size, self.hidden_size)
        self.critic = GNNValue(self.input_size, self.hidden_size)
        self.parser = parser
        
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
        action, log_prob = self.actor(x)

        # critic: estimates V(s_t)
        value = self.critic(x)
        return action, log_prob, value
    
    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state
    
    def select_action(self, obs):
        action, log_prob, value = self.forward(obs)
        
        self.saved_actions.append(SavedAction(log_prob, value))
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
    
    def test_agent(self, test_episodes, env, cplexpath, matching_steps, agent_name):
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
            if self.sim == 'sumo':
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

                o = self.parser.parse_obs(obs)

                action_rl = self.select_action(o, deterministic=True)
                actions.append(action_rl)

                desired_acc = {env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + env.tstep)) for i in range(len(env.region))}

                reb_action = solveRebFlow(env, f'scenario_lux/{agent_name}', desired_acc, cplexpath)

                # Take action in environment
                try:
                    _, rebreward, done, info = env.reb_step(reb_action)
                except Exception as e:
                    print(f"FatalTraCIError during rebalancing step: {e}")
                    if self.sim == 'sumo':
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
            if done and self.sim == 'sumo':
                traci.close()

        return (
            np.mean(episode_reward),
            np.mean(episode_served_demand),
            np.mean(episode_rebalancing_cost),
        )
    

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
