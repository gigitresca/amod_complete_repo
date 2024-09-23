# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:09:46 2020
@author: yangk
"""
from collections import defaultdict
import numpy as np
import subprocess
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import re
from tqdm import trange
from src.misc.utils import mat2str


class MPC:
    """
    Model Predictive Control (MPC) for AMoD
    """
    def __init__(self, env, cplexpath=None, platform=None, T = 20):
        """
        Initialize the MPC object.
        - env: The AMoD environment object that contains various attributes and methods.
        - cplexpath: The path to the CPLEX executable.
        - platform: The platform to run the CPLEX executable.
        - T: The time horizon for the MPC.
        """
        self.env = env 
        self.T = T
        self.CPLEXPATH = cplexpath
        self.platform = platform
        if self.CPLEXPATH is None:
            self.CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"

    def MPC_exact(self):
        """
        Exact MPC for AMoD
        """
        sim = self.env.cfg.name
        t = self.env.time
        if sim == "sumo":
            tstep = self.env.tstep
            reservations = traci.person.getTaxiReservations(3)
            # Assign actual demand (for passengers not previously served)
            flows = list()
            demandAttr = list()
            for trip in reservations:
                persons = trip.persons[0]
                waiting_time = traci.person.getWaitingTime(persons)
                if waiting_time > self.env.max_waiting_time * 60:
                    traci.person.remove(persons)
                    persons = ''
                if not persons:
                    continue
                o = int(persons[(persons.find('o') + 1):persons.find('d')])
                d = int(persons[(persons.find('d') + 1):persons.find('#')])
                if (o, d, self.env.demand_time[o, d][t], self.env.price[o, d][t]) in flows:
                    idx = flows.index((o, d, self.env.demand_time[o, d][t], self.env.price[o, d][t]))
                    flow = demandAttr[idx][3] + 1
                    demandAttr[idx] = (o, d, t, flow, self.env.demand_time[(o, d)][t], self.env.price[o, d][t])
                else:
                    flows.append((o, d, self.env.demand_time[o, d][t], self.env.price[o, d][t]))
                    demandAttr.append((o, d, t, 1, self.env.demand_time[o, d][t], self.env.price[o, d][t]))
            # Future demand
            demandAttr.extend([(i,j,tt,self.env.demand[i,j][tt], self.env.demand_time[i,j][tt], self.env.price[i,j][tt]) for i,j in self.env.demand for tt in range(t+tstep, min(t+self.T, self.env.duration), tstep) if self.env.demand[i,j][tt]>1e-3])
            accTuple = [(n, self.env.acc[n][t + tstep]) for n in self.env.acc]
            daccTuple = [(n, tt, self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t, min(t + self.T, self.env.duration))]
        else:
            demandAttr = [(i, j, tt, self.env.demand[i, j][tt], self.env.demandTime[i, j][tt], self.env.price[i, j][tt]) for i, j in self.env.demand for tt in range(t, t + self.T) if self.env.demand[i, j][tt] > 1e-3]
            accTuple = [(n, self.env.acc[n][t]) for n in self.env.acc]
            daccTuple = [(n, tt, self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t, t + self.T)]

        edgeAttr = [(i,j,self.env.reb_time[i,j][t]) for i,j in self.env.edges]
        modPath = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
        MPCPath = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/MPC/scenario_lux/exact/'
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + 'data_{}.dat'.format(t)
        resfile = MPCPath + 'res_{}.dat'.format(t)
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('t0='+str(t)+';\r\n')
            file.write('T='+str(self.T)+';\r\n')
            file.write('beta='+str(self.env.beta)+';\r\n')
            file.write('demandAttr='+mat2str(demandAttr)+';\r\n')
            file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
            file.write('daccAttr='+mat2str(daccTuple)+';\r\n')
        
        modfile = modPath+'MPC.mod'
        my_env = os.environ.copy()
        if self.platform == None:
            my_env["LD_LIBRARY_PATH"] = self.CPLEXPATH
        else:
            my_env["DYLD_LIBRARY_PATH"] = self.CPLEXPATH
        out_file =  MPCPath + 'out_{}.dat'.format(t)
        with open(out_file,'w') as output_f:
            subprocess.check_call([self.CPLEXPATH+"oplrun", modfile,datafile],stdout=output_f,env=my_env)
        output_f.close()
        paxFlow = defaultdict(float)
        rebFlow = defaultdict(float)
        with open(resfile,'r', encoding="utf8") as file:
            for row in file:
                item = row.replace('e)',')').strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,j,f1,f2 = v.split(',')
                        f1 = float(re.sub('[^0-9e.-]','', f1))
                        f2 = float(re.sub('[^0-9e.-]','', f2))
                        paxFlow[int(i),int(j)] = float(f1)
                        rebFlow[int(i),int(j)] = float(f2)
        paxAction = [paxFlow[i,j] if (i,j) in paxFlow else 0 for i,j in self.env.edges]
        rebAction = [rebFlow[i,j] if (i,j) in rebFlow else 0 for i,j in self.env.edges]
        return paxAction,rebAction

    def test(self, num_episodes, env):
        """
        for testing MPC
        - num_episodes: An integer representing the number of episodes to run the test.
        - env: The AMoD environment object that contains various attributes and methods.
        """
        sim = env.cfg.name
        if sim == "sumo":
            # traci.close(wait=False)
            os.makedirs(f'saved_files/sumo_output/{env.cfg.city}/', exist_ok=True)
            matching_steps = int(env.cfg.matching_tstep * 60 / env.cfg.sumo_tstep)  # sumo steps between each matching
            if env.scenario.is_meso:
                matching_steps -= 1

            sumo_cmd = [
                "sumo", "--no-internal-links", "-c", env.cfg.sumocfg_file,
                "--step-length", str(env.cfg.sumo_tstep),
                "--device.taxi.dispatch-algorithm", "traci",
                "-b", str(env.cfg.time_start * 60 * 60), "--seed", "10",
                "-W", 'true', "-v", 'false',
            ]
            assert os.path.exists(env.cfg.sumocfg_file), "SUMO configuration file not found!"
        epochs = trange(num_episodes)
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        inflows = []
        for i_episode in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            inflow = np.zeros(env.nregion)
            done = False
            if sim =='sumo':
                traci.start(sumo_cmd)
            obs, rew = env.reset()
            eps_reward += rew
            eps_served_demand += rew
            while not done:
                pax_action, reb_action = self.MPC_exact()
                obs, rew, done, info = env.step(reb_action, reb_action)
                for k in range(len(env.edges)):
                    i,j = env.edges[k]
                    inflow[j] += reb_action[k]
                eps_reward += rew
                eps_served_demand += info["profit"]
                eps_rebalancing_cost += info["rebalancing_cost"]
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)
            inflows.append(inflow)
            epochs.set_description(f"Test Episode {i_episode+1} | Reward: {eps_reward:.2f} | ServedDemand: {eps_served_demand:.2f} | Reb. Cost: {eps_rebalancing_cost:.2f}")
        return episode_reward, episode_served_demand, episode_rebalancing_cost, inflows
