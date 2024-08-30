from __future__ import print_function
import torch
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import os
import platform
import numpy as np
import argparse

from src.envs.sim.amod_env import Scenario, AMoD
from tqdm import trange

parser = argparse.ArgumentParser(description='MPC')

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
parser.add_argument('--sumo_tstep', type=int, default=1, metavar='S',
                    help='sumo time step (default: 1 s)')
parser.add_argument('--max_waiting_time', type=int, default=7200, metavar='S',
                    help='maximum passengers waiting time for a ride (default: 10 min)')
parser.add_argument('--beta', type=float, default=1, metavar='S',
                    help='cost of rebalancing (default: 1)')
parser.add_argument('--num_regions', type=int, default=8, metavar='S',
                    help='Number of regions for spatial aggregation (default: 8)')
parser.add_argument('--policy_name', type=str, default='noreb',
                    help='Pretrained agent selection for agent evaluation (default: uniform)')
parser.add_argument('--max_episodes', type=int, default=1, metavar='N',
                    help='number of episodes to train agent (default: 1)')
parser.add_argument('--random_od', default=False, action='store_true',
                    help='Demand aggregated in the centers of the regions (default: False)')
parser.add_argument('--acc_init', type=int, default=90, metavar='S',
                    help='Initial number of taxis per region (default: 90)')

args = parser.parse_args()
if platform.system() == 'Windows':
    cplexpath = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/opl/bin/x64_win64/"
elif platform.system() == 'Mac':
    cplexpath = "/Applications/CPLEX_Studio1210/opl/bin/x86-64_osx/"
elif platform.system() == 'Linux':
    cplexpath = "/opt/ibm/ILOG/CPLEX_Studio2211/opl/bin/x86-64_linux/"
matching_steps = int(args.matching_tstep * 60 / args.sumo_tstep)  # sumo steps between each matching
aggregated_demand = not args.random_od

# Define SUMO scenario
scenario_path = 'src/envs/data/LuSTScenario/'
sumocfg_file = 'dua_meso.static.sumocfg'
net_file = os.path.join(scenario_path, 'input/lust_meso.net.xml')
demand_file = f'src/envs/data/scenario_lux{args.num_regions}.json'
os.makedirs('saved_files/sumo_output/scenario_lux/', exist_ok=True)
sumo_cmd = [
        "sumo", "--no-internal-links", "-c", os.path.join(scenario_path, sumocfg_file),
        "--step-length", str(args.sumo_tstep),
        "--device.taxi.dispatch-algorithm", "traci",
        "--summary-output", "saved_files/sumo_output/scenario_lux/" + args.policy_name + "_dua_meso.static.summary.xml",
        "--tripinfo-output", "saved_files/sumo_output/scenario_lux/" + args.policy_name + "_dua_meso.static.tripinfo.xml",
        "--tripinfo-output.write-unfinished", "true",
        "--person-summary-output", "saved_files/sumo_output/scenario_lux/" + args.policy_name + "_dua_meso.static.person.xml",
        "--log", "saved_files/sumo_output/scenario_lux/dua_meso.static.log",
        "-b", str(args.time_start * 60 * 60), "--seed", str(args.seed),
        "-W", 'true', "-v", 'false'
    ]
# Define AMoD Simulator Environment

scenario = Scenario(num_cluster=args.num_regions, json_file=demand_file, aggregated_demand=aggregated_demand,
                    sumo_net_file=net_file, acc_init=args.acc_init, sd=args.seed, demand_ratio=args.demand_ratio,
                    time_start=args.time_start, time_horizon=args.time_horizon, duration=args.duration,
                    tstep=args.matching_tstep, max_waiting_time=args.max_waiting_time)

env = AMoD(scenario, beta=args.beta)
if 'meso' in net_file:
    matching_steps -= 1     # In the meso setting one step is done within the reb_step
epochs = trange(args.max_episodes)     # epoch iterator
# Initialize lists for logging
log = list()
tot_demand = list()
for i in range(args.max_episodes):
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
        # take matching step
        obs, paxreward, done, info = env.pax_step(CPLEXPATH=cplexpath, PATH='uniform/scenario_lux_test')
        episode_reward += paxreward
        # No rebalancing step
        reb_action = [0] * len(env.edges)
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
    tot_demand.append(sum([env.demand[i, j][t] for i, j in env.demand for t in range(0, args.duration * 60, args.matching_tstep)]))
    epochs.set_description(f"Episode {episode+1} | Reward: {episode_reward:.2f} | Revenue: {episode_revenue:.2f} | ServedDemand: {(episode_served_demand / tot_demand[-1])*100} % | Reb. Cost: {episode_rebalancing_cost:.2f} | Op. Cost: {episode_operating_cost:.2f} | Reb. Veh: {episode_rebalanced_vehicles}")

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
for i in range(args.max_episodes):
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
print(f"No reb mean: Reward {log_stat['test_reward']['mean']:.2f}, Revenue {log_stat['test_revenue']['mean']:.2f},Served demand {log_stat['test_served_demand']['mean']:.2f}, Rebalancing Cost {log_stat['test_reb_cost']['mean']:.2f}, Operational Cost {log_stat['test_op_cost']['mean']:.2f}, Rebalanced Vehicles {log_stat['test_reb_vehicles']['mean']:.2f}")
print(f"No reb std: Reward {log_stat['test_reward']['std']:.2f}, Revenue {log_stat['test_revenue']['std']:.2f},Served demand {log_stat['test_served_demand']['std']:.2f}, Rebalancing Cost {log_stat['test_reb_cost']['std']:.2f}, Operational Cost {log_stat['test_op_cost']['std']:.2f}, Rebalanced Vehicles {log_stat['test_reb_vehicles']['std']:.2f}")
os.makedirs('saved_files/baseline_policies/scenario_lux/', exist_ok=True)
torch.save({'log': log, 'log_stat': log_stat}, f'./saved_files/baseline_policies/scenario_lux/{args.policy_name}_test.pth')
