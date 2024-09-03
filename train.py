import hydra
from omegaconf import DictConfig
import os 
import torch

@hydra.main(version_base=None, config_path="config", config_name="config")
def setup(cfg: DictConfig):
   
    # Import simulator module based on the configuration
    simulator_name = cfg.simulator.name
    if simulator_name == "sumo":
        from src.sim.sumo_env import Scenario, AMoD
        demand_file = f'src/envs/data/scenario_lux{cfg.num_regions}.json'
        aggregated_demand = not cfg.random_od
        scenario_path = 'src/envs/data/LuSTScenario/'
        net_file = os.path.join(scenario_path, 'input/lust_meso.net.xml')
        scenario = Scenario(num_cluster=cfg.num_regions, json_file=demand_file, aggregated_demand=aggregated_demand,
                    sumo_net_file=net_file, acc_init=cfg.acc_init, sd=cfg.seed, demand_ratio=cfg.demand_ratio,
                    time_start=cfg.time_start, time_horizon=cfg.time_horizon, duration=cfg.duration,
                    tstep=cfg.matching_tstep, max_waiting_time=cfg.max_waiting_time)
        env = AMoD(scenario, beta=cfg.beta)

    elif simulator_name == "macro":
        scenario = Scenario(json_file="data/scenario_nyc4x4.json", sd=cfg.seed, demand_ratio=cfg.demand_ratio, json_hr=cfg.json_hr, json_tstep=cfg.json_tsetp)
        env = AMoD(scenario, beta = 0.5)
    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")

    # Import model module based on the configuration
    model_name = cfg.model.name
    cfg.cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cfg.cuda else "cpu")

    if cfg.model.name == "sac":
        from src.algos import SAC
        model = SAC(
            env=env,
            input_size=21,
            hidden_size=cfg.hidden_size,
            p_lr=cfg.p_lr,
            q_lr=cfg.q_lr,
            alpha=cfg.alpha,
            batch_size=cfg.batch_size,
            use_automatic_entropy_tuning=cfg.auto_entropy,
            clip=cfg.clip,
            critic_version=cfg.critic_version,
        ).to(device)
        
    elif model_name == "a2c":
        from src.algos import A2C
        model = A2C(env=env, input_size=21).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

if __name__ == "__main__":
    setup()