from __future__ import print_function
import argparse
import numpy as np
import torch
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci

from tqdm import trange
from src.envs.sim.amod_env import Scenario, AMoD
from src.algos.a2c import A2C
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
from datetime import date


parser = argparse.ArgumentParser(description='A2C-GNN')

# Simulator parameters
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=float, default=0.8, metavar='S',
                    help='demand_ratio (default: 0.8)')
parser.add_argument('--time_start', type=int, default=7, metavar='S',
                    help='simulation start time in hours (default: 7 hr)')
parser.add_argument('--duration', type=int, default=2, metavar='S',
                    help='episode duration in hours (default: 2 hr)')
parser.add_argument('--time_horizon', type=int, default=10, metavar='S',
                    help='matching steps in the future for demand and arriving vehicle forecast (default: 10 min)')
parser.add_argument('--matching_tstep', type=int, default=1, metavar='S',
                    help='minutes per timestep (default: 1 min)')
parser.add_argument('--max_waiting_time', type=int, default=10, metavar='S',
                    help='maximum passengers waiting time for a ride (default: 10 min)')
parser.add_argument('--beta', type=float, default=1, metavar='S',   ########## MODIFIED THE DEFUAULT VALUE
                    help='cost of rebalancing (default: 1)')
parser.add_argument('--num_regions', type=int, default=8, metavar='S',
                    help='Number of regions fro spatial aggregation (default: 8)')
parser.add_argument('--random_od', default=False, action='store_true',
                    help='Demand aggregated in the centers of the regions (default: False)')
parser.add_argument('--acc_init', type=int, default=90, metavar='S',
                    help='Initial number of taxis per region (default: 90)')
# Model parameters
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--agent_name', type=str, default=date.today().strftime("%Y%m%d")+'_a2c_gnn',
                    help='Pretrained agent selection for agent evaluation (default:'+date.today().strftime("%Y%m%d")+'_a2c_gnn')
parser.add_argument('--cplexpath', type=str, default='/opt/ibm/ILOG/CPLEX_Studio2211/opl/bin/x86-64_linux/',
                    help='defines directory of the CPLEX installation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=5000, metavar='N',
                    help='number of episodes to train agent (default: 16k)')
parser.add_argument('--max_steps', type=int, default=120, metavar='N',
                    help='number of steps per episode (default: T=120)')
parser.add_argument('--no-cuda', type=bool, default=True,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
aggregated_demand = not args.random_od

# Define SUMO scenario
scenario_path = 'src/envs/data/LuSTScenario/'
sumocfg_file = 'dua_meso.static.sumocfg'
net_file = os.path.join(scenario_path, 'input/lust_meso.net.xml')
os.makedirs('saved_files/sumo_output/scenario_lux/', exist_ok=True)
if args.test:
    sumo_cmd = [
        "sumo", "--no-internal-links", "-c", os.path.join(scenario_path, sumocfg_file),
        "--device.taxi.dispatch-algorithm", "traci",
        "--summary-output", "saved_files/sumo_output/scenario_lux/" + args.agent_name + "_dua_meso.static.summary.xml",
        "--tripinfo-output", "saved_files/sumo_output/scenario_lux/" + args.agent_name + "_dua_meso.static.tripinfo.xml",
        "--tripinfo-output.write-unfinished", "true",
        "--log", "saved_files/sumo_output/scenario_lux/dua_meso.static.log",
        "-b", str(args.time_start * 60 * 60), "--seed", str(args.seed),
        "-W", 'true', "-v", 'false'
    ]
else:
    sumo_cmd = [
        "sumo", "--no-internal-links", "-c", os.path.join(scenario_path, sumocfg_file),
        "--device.taxi.dispatch-algorithm", "traci",
        "-b", str(args.time_start * 60 * 60), "--seed", "10",
        "-W", 'true', "-v", 'false',
    ]
demand_file = f'src/envs/data/scenario_lux{args.num_regions}.json'

# Define AMoD Simulator Environment
scenario = Scenario(num_cluster=args.num_regions, json_file=demand_file, aggregated_demand=aggregated_demand,
                    sumo_net_file=net_file, acc_init=args.acc_init, sd=args.seed, demand_ratio=args.demand_ratio,
                    time_start=args.time_start, time_horizon=args.time_horizon, duration=args.duration,
                    tstep=args.matching_tstep, max_waiting_time=args.max_waiting_time)
env = AMoD(scenario, beta=args.beta)
# Initialize A2C-GNN
model = A2C(env=env, input_size=21).to(device)
# Define the sumo steps between each agent decision
matching_steps = int(args.matching_tstep * 60)  # sumo steps between each matching
if 'meso' in net_file:
    matching_steps -= 1     # In the meso setting one step is done within the reb_step

if not args.test:
    #######################################
    #############Training Loop#############
    #######################################

    #Initialize lists for logging
    log = {'train_reward': [], 
           'train_served_demand': [], 
           'train_reb_cost': [],
           'train_reb_vehicles': [],
           'train_policy_losses': [],
           'train_value_losses': []}

    train_episodes = args.max_episodes #set max number of training episodes
    T = args.max_steps #set episode length
    epochs = trange(train_episodes)     # epoch iterator
    best_reward = -np.inf   # set best reward
    model.train()   # set model in train mode

    for i_episode in epochs:
        # Initialize the sumo simulation
        traci.start(sumo_cmd)
        # Initialize the reward
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        episode_rebalanced_vehicles = 0
        # Reset the environment
        obs = env.reset()  # initialize environment
        if 'meso' in net_file:
            try:
                traci.simulationStep()
            except Exception as e:
                print(f"FatalTraCIError during initial step: {e}")
                traci.close()
                break
        for step in range(T):
            sumo_step = 0
            try:
                while sumo_step < matching_steps:
                    traci.simulationStep()
                    sumo_step += 1
            except Exception as e:
                print(f"FatalTraCIError during matching steps: {e}")
                traci.close()
                break
            # take matching step (Step 1 in paper)
            obs, paxreward, done, info = env.pax_step(CPLEXPATH=args.cplexpath, PATH=f'a2c/scenario_lux/{args.agent_name}')
            episode_reward += paxreward
            # use GNN-RL policy (Step 2 in paper)
            action_rl = model.select_action(obs)
            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desired_acc = {env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time+env.tstep))for i in range(len(env.region))}
            # solve minimum rebalancing distance problem (Step 3 in paper)
            reb_action = solveRebFlow(env, f'a2c/scenario_lux/{args.agent_name}', desired_acc, args.cplexpath)
            # Take action in environment
            try:
                new_obs, rebreward, done, info = env.reb_step(reb_action)
            except Exception as e:
                print(f"FatalTraCIError during rebalancing step: {e}")
                traci.close()
                break
            episode_reward += rebreward
            # Store the transition in memory
            model.rewards.append(paxreward + rebreward)
            # track performance over episode
            episode_served_demand += info['served_demand']
            episode_rebalancing_cost += info['rebalancing_cost']
            episode_rebalanced_vehicles += info['rebalanced_vehicles']
            # stop episode if terminating conditions are met
            if done:
                traci.close()
                break

        # Training step only if the episode was completed
        if step < (args.duration * 60) / args.matching_tstep - 1:
            epochs.set_description(f"Episode {i_episode + 1} | Not completed")
            continue
        # perform on-policy backprop
        p_loss, v_loss = model.training_step()
        log['train_policy_losses'].extend(p_loss)
        log['train_value_losses'].extend(v_loss)

        # Send current statistics to screen
        epochs.set_description(f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}  | Reb. Veh: {episode_rebalanced_vehicles:.2f}")
        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(path=f"./{args.directory}/ckpt/scenario_lux/{args.agent_name}.pth")
            best_reward = episode_reward
        # Log KPIs
        log['train_reward'].append(episode_reward)
        log['train_served_demand'].append(episode_served_demand)
        log['train_reb_cost'].append(episode_rebalancing_cost)
        log['train_reb_vehicles'].append(episode_rebalanced_vehicles)
        model.log(log, path=f"./{args.directory}/rl_logs/scenario_lux/{args.agent_name}.pth")
else:
    # Load pre-trained model
    model.load_checkpoint(path=f"./{args.directory}/ckpt/scenario_lux/{args.agent_name}.pth")
    test_episodes = args.max_episodes #set max number of training episodes
    T = args.max_steps #set episode length
    epochs = trange(test_episodes) #epoch iterator
    # Initialize lists for logging
    log = list()
    tot_demand = list()
    for i in range(test_episodes):
        log_dict = {
            'test_reward': [],
            'test_served_demand': [],
            'test_reb_cost': [],
            'test_op_cost': [],
            'test_reb_vehicles': [],
            'test_revenue': []
        }
        log.append(log_dict)

    for episode in epochs:
        # Initialize the sumo simulation
        traci.start(sumo_cmd)
        # Initialize the reward
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        episode_operating_cost = 0
        episode_rebalanced_vehicles = 0
        episode_revenue = 0
        done = False
        # Reset the environment
        obs = env.reset()
        if 'meso' in net_file:
            traci.simulationStep()
        while not done:
            sumo_step = 0
            while sumo_step < matching_steps:
                traci.simulationStep()
                sumo_step += 1
            # take matching step (Step 1 in paper)
            obs, paxreward, done, info = env.pax_step(CPLEXPATH=args.cplexpath, PATH=f'a2c/scenario_lux/{args.agent_name}')
            episode_reward += paxreward
            # use GNN-RL policy (Step 2 in paper)
            action_rl = model.select_action(obs)
            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desired_acc = {env.region[i]: int(action_rl[i] * dictsum(env.acc, env.time + env.tstep)) for i in range(len(env.region))}
            # solve minimum rebalancing distance problem (Step 3 in paper)
            reb_action = solveRebFlow(env, f'a2c/scenario_lux/{args.agent_name}', desired_acc, args.cplexpath)
            # Take action in environment
            new_obs, rebreward, done, info = env.reb_step(reb_action)
            episode_reward += rebreward
            # track performance over episode
            episode_served_demand += info['served_demand']
            episode_rebalancing_cost += info['rebalancing_cost']
            episode_operating_cost += info['operating_cost']
            episode_rebalanced_vehicles += info['rebalanced_vehicles']
            episode_revenue += info['revenue']
            # Log KPIs
            log[episode]['test_reward'].append(episode_reward)
            log[episode]['test_served_demand'].append(episode_served_demand)
            log[episode]['test_reb_cost'].append(episode_rebalancing_cost)
            log[episode]['test_op_cost'].append(episode_operating_cost)
            log[episode]['test_reb_vehicles'].append(episode_rebalanced_vehicles)
            log[episode]['test_revenue'].append(episode_revenue)
            # stop episode if terminating conditions are met
            if done:
                traci.close()
                break
        # Send current statistics to screen
        tot_demand.append(sum([env.demand[i, j][t] for i, j in env.demand for t in range(0, args.duration*60)]))
        epochs.set_description(f"Episode {episode + 1} | Reward: {episode_reward:.2f} | Revenue: {episode_revenue:.2f} | ServedDemand: {(episode_served_demand / tot_demand[-1]) * 100} % | Reb. Cost: {episode_rebalancing_cost:.2f} | Op. Cost: {episode_operating_cost:.2f} | Reb. Veh: {episode_rebalanced_vehicles}")
    # Test statistic calculation
    log_stat = {
        'test_reward': {'mean', 'std'},
        'test_served_demand': {'mean', 'std'},
        'test_reb_cost': {'mean', 'std'},
        'test_op_cost': {'mean', 'std'},
        'test_reb_vehicles': {'mean', 'std'},
        'test_revenue': {'mean', 'std'}
    }
    reward = []
    served_demand = []
    reb_cost = []
    op_cost = []
    reb_vehicles = []
    revenue = []
    for i in range(test_episodes):
        episode = log[i]
        reward.append(episode['test_reward'][-1])
        served_demand.append(episode['test_served_demand'][-1] / tot_demand[i])
        reb_cost.append(episode['test_reb_cost'][-1])
        op_cost.append(episode['test_op_cost'][-1])
        reb_vehicles.append(episode['test_reb_vehicles'][-1])
        revenue.append(episode['test_revenue'][-1])

    log_stat = {
        'test_reward': {'mean': np.mean(np.array(reward)), 'std': np.std(np.array(reward))},
        'test_served_demand': {'mean': np.mean(np.array(served_demand)), 'std': np.std(np.array(served_demand))},
        'test_reb_cost': {'mean': np.mean(np.array(reb_cost)), 'std': np.std(np.array(reb_cost))},
        'test_op_cost': {'mean': np.mean(np.array(op_cost)), 'std': np.std(np.array(op_cost))},
        'test_reb_vehicles': {'mean': np.mean(np.array(reb_vehicles)), 'std': np.std(np.array(reb_vehicles))},
        'test_revenue': {'mean': np.mean(np.array(revenue)), 'std': np.std(np.array(revenue))},
    }
    print(f"Uniform mean: Reward {log_stat['test_reward']['mean']:.2f}, Revenue {log_stat['test_revenue']['mean']:.2f},Served demand {log_stat['test_served_demand']['mean']:.2f}, Rebalancing Cost {log_stat['test_reb_cost']['mean']:.2f}, Operational Cost {log_stat['test_op_cost']['mean']:.2f}, Rebalanced Vehicles {log_stat['test_reb_vehicles']['mean']:.2f}")
    print(f"Uniform std: Reward {log_stat['test_reward']['std']:.2f}, Revenue {log_stat['test_revenue']['std']:.2f},Served demand {log_stat['test_served_demand']['std']:.2f}, Rebalancing Cost {log_stat['test_reb_cost']['std']:.2f}, Operational Cost {log_stat['test_op_cost']['std']:.2f}, Rebalanced Vehicles {log_stat['test_reb_vehicles']['std']:.2f}")
    torch.save({'log': log, 'log_stat': log_stat}, f'./saved_files/rl_logs/scenario_lux/{args.agent_name}_test.pth')
