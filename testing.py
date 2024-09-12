import hydra
from omegaconf import DictConfig
import os 
import torch
import json
import numpy as np

def setup_sumo(cfg):
    from src.envs.sim.sumo_env import Scenario, AMoD, GNNParser

    cfg = cfg.simulator
    demand_file = f'src/envs/data/scenario_lux{cfg.num_regions}.json'
    aggregated_demand = not cfg.random_od
    scenario_path = 'src/envs/data/LuSTScenario/'
    net_file = os.path.join(scenario_path, 'input/lust_meso.net.xml')

    scenario = Scenario(num_cluster=cfg.num_regions, json_file=demand_file, aggregated_demand=aggregated_demand,
                sumo_net_file=net_file, acc_init=cfg.acc_init, sd=cfg.seed, demand_ratio=cfg.demand_ratio,
                time_start=cfg.time_start, time_horizon=cfg.time_horizon, duration=cfg.duration,
                tstep=cfg.matching_tstep, max_waiting_time=cfg.max_waiting_time)
    env = AMoD(scenario, beta=cfg.beta)
    parser = GNNParser(env, T=cfg.time_horizon, json_file=demand_file)
    return env, parser
    
def setup_macro(cfg):
    from src.envs.sim.macro_env import Scenario, AMoD, GNNParser
    with open("src/envs/data/macro/calibrated_parameters.json", "r") as file:
        calibrated_params = json.load(file)
    cfg.simulator.cplexpath = cfg.model.cplexpath
    cfg = cfg.simulator
    city = cfg.city
    scenario = Scenario(
    json_file=f"src/envs/data/macro/scenario_{city}.json",
    demand_ratio=calibrated_params[city]["demand_ratio"],
    json_hr=calibrated_params[city]["json_hr"],
    sd=cfg.seed,
    #json_tstep=cfg.json_tsetp,
    json_tstep=4,
    tf=cfg.max_steps,
    )
    env = AMoD(scenario, cfg = cfg, beta = calibrated_params[city]["beta"])
    parser = GNNParser(env, T=cfg.time_horizon, json_file=f"src/envs/data/macro/scenario_{city}.json")
    return env, parser

def setup_model(cfg, env, parser, device):
    model_name = cfg.model.name

    if model_name == "sac":
        from src.algos.sac import SAC
        return SAC(env=env, input_size=cfg.model.input_size, cfg=cfg.model, parser=parser).to(device)
    
    elif model_name == "a2c":
        from src.algos.a2c import A2C
        return A2C(env=env, input_size=cfg.model.input_size, parser=parser).to(device)
    
    elif model_name == "equal_distribution":
        from src.algos.ed import EqualDistribution
        return EqualDistribution(cplexpath=cfg.model.cplexpath, directory=cfg.model.directory)
    
    elif model_name =="plus_one": 
        from src.algos.plus_one import PlusOneBaseline
        return PlusOneBaseline(cplexpath=None)
    
    elif model_name == "no_rebalancing":
        from src.algos.no_reb import NoRebalanceBaseline
        return NoRebalanceBaseline()
    
    else:
        raise ValueError(f"Unknown model or baseline: {model_name}")

def setup_net(cfg):
    if cfg.model.use_LSTM:
        from src.nets.actor import GNNActorLSTM as GNNActor
        from src.nets.critic import GNNCriticLSTM as GNNCritic
    else:
        from src.nets.actor import GNNActor
        from src.nets.critic import GNNCritic

        models = {
        "actor": GNNActor,
        "critic": GNNCritic,
        }
    return models

import torch
import numpy as np
from hydra import initialize, compose

def test(config):

    with initialize(config_path="src/config"):
        cfg = compose(config_name="config", overrides= [f"{key}={value}" for key, value in config.items()])  # Load the configuration
        
    # Import simulator module based on the configuration
    simulator_name = cfg.simulator.name
    if simulator_name == "sumo":
        env, parser = setup_sumo(cfg)
    elif simulator_name == "macro":
        env, parser = setup_macro(cfg)
    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")
    
    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = setup_model(cfg, env, parser, device)

    print(f'Testing model {cfg.model.name} on {cfg.simulator.name} environment')
    episode_reward, episode_served_demand, episode_rebalancing_cost = model.test(10, env)

    print('Mean Episode Profit ($): ', np.mean(episode_reward))
    print('Mean Episode Served Demand($): ', np.mean(episode_served_demand))
    print('Mean Episode Rebalancing Cost($): ', np.mean(episode_rebalancing_cost))


    no_reb_reward = 27592.241758477943
    no_reb_demand = 1599.6
    no_reb_cost = 0.0
    mean_reward = np.mean(episode_reward)
    mean_served_demand = np.mean(episode_served_demand)
    mean_rebalancing_cost = np.mean(episode_rebalancing_cost)


    labels = ['Reward', 'Served Demand', 'Rebalancing Cost']
    rl_means = [mean_reward, mean_served_demand, mean_rebalancing_cost]

    no_control = [no_reb_reward, no_reb_demand, no_reb_cost]

    import matplotlib.pyplot as plt
    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, rl_means, width, label=cfg.model.name, color='tab:blue', capsize=5)
    rects2 = ax.bar(x + width/2, no_control, width, label='No Control', color='tab:orange')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Metrics')
    ax.set_ylabel('$')
    ax.set_title(f'Comparison of {cfg.model.name} vs No Control')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Function to add value labels on top of bars
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Adding value labels to each bar
    add_value_labels(rects1)
    add_value_labels(rects2)

    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()
    

@hydra.main(version_base=None, config_path="src/config/", config_name="config")
def main(cfg: DictConfig):
   
    # Import simulator module based on the configuration
    simulator_name = cfg.simulator.name
    if simulator_name == "sumo":
        env, parser = setup_sumo(cfg)

    elif simulator_name == "macro":
        env, parser = setup_macro(cfg)
    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")

    use_cuda = not cfg.model.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = setup_model(cfg, env, parser, device)

    print('Testing...')
    episode_reward, episode_served_demand, episode_rebalancing_cost = model.test(10, env)

    print('Mean Episode Profit ($): ', np.mean(episode_reward), 'Std Episode Reward: ', np.std(episode_reward))
    print('Mean Episode Served Demand($): ', np.mean(episode_served_demand), 'Std Episode Served Demand: ', np.std(episode_served_demand))
    print('Mean Episode Rebalancing Cost($): ', np.mean(episode_rebalancing_cost), 'Std Episode Rebalancing Cost: ', np.std(episode_rebalancing_cost))

    ##TODO: ADD VISUALIZATION

if __name__ == "__main__":
    main()
    