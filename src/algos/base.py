class BaseAlgorithm:
    def __init__(self):
        """
        Base class for baseline algorithms.
        """
        pass
        
    def select_action(self, env):
        """
        This method should be overridden by derived classes to return the rebalance action.

        Possible variables to get from the environment:

        env.nregion: number of regions
        len(env.edges): number of edges
        env.acc: accumulated number of vehicles in each region
        env.time: current time step in environment 
        env.G.edges[i, j]["time"]: travel time from region i to region j
        env.scenario.demand_input[i, j][t]: demand forecast from region i to region j at time t
        """

        raise NotImplementedError("The select_action method must be implemented by subclasses.")

    def test(self, num_episodes, env):
        epochs = range(num_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []

        for _ in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            
            done = False
            obs, rew = env.reset() 
            eps_reward += rew
            
            while not done:

                reb_action = self.select_action(env)
            
                obs, rew, done, info = env.step(reb_action)
 
                eps_reward += rew
                eps_served_demand += info["served_demand"]
                eps_rebalancing_cost += info["rebalancing_cost"]
    
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)

        return episode_reward, episode_served_demand, episode_rebalancing_cost
