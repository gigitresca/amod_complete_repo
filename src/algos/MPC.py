# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 22:09:46 2020
@author: yangk
"""
from collections import defaultdict
import numpy as np
import subprocess
import os
import traci
import networkx as nx
from src.misc.utils import mat2str
from copy import deepcopy
import re

class MPC:
    def __init__(self, env, CPLEXPATH=None, platform = None, T = 20):
        self.env = env 
        self.T = T
        self.CPLEXPATH = CPLEXPATH
        if self.CPLEXPATH is None:
            self.CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        self.platform = platform

    def MPC_exact(self):

        ###################
        # ADDED

        tstep = self.env.tstep
        t = self.env.time
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
        ############################



        accTuple = [(n,self.env.acc[n][t+tstep]) for n in self.env.acc]
        daccTuple = [(n,tt,self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t,min(t+self.T, self.env.duration))]
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
    
    def bi_level_matching(self):
        tstep = self.env.tstep
        t = self.env.time
        demandAttr = [(i,j,tt,self.env.demand[i,j][tt], self.env.demand_time[i,j][tt], self.env.price[i,j][tt]) for i,j in self.env.demand for tt in range(t,min(t+self.T, self.env.duration)) if self.env.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.env.acc[n][t]) for n in self.env.acc]
        daccTuple = [(n,tt,self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t,min(t+self.T, self.env.duration))]
        edgeAttr = [(i,j,self.env.reb_time[i,j][t]) for i,j in self.env.G.edges]
        modPath = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
        MPCPath = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/MPC/scenario_lux/bi-level-matching/'
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
           
        modfile = modPath+'bi-level-matching.mod'
        
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
        return rebAction
        
    def bi_level_rebalancing(self):
        tstep = self.env.tstep
        t = self.env.time
        demandAttr = [(i,j,tt,self.env.demand[i,j][tt], self.env.price[i,j][tt]) for i,j in self.env.demand for tt in range(t,min(t+self.T, self.env.duration)) if self.env.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.env.acc[n][t]) for n in self.env.acc]
        daccTuple = [(n,tt,self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t,min(t+self.T, self.env.duration))]
        edgeAttr = [(i,j,self.env.G.edges[i,j]['time']) for i,j in self.env.G.edges]
        modPath = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
        MPCPath = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/MPC/scenario_lux/bi-level-rebalancing/'
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
           
        modfile = modPath+'bi-level-rebalancing.mod'
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
        desiredAcc = defaultdict(float)
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
                elif item[0] == 'desiredAcc':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,f = v.split(',')
                        f = float(re.sub('[^0-9e.-]','', f))
                        desiredAcc[int(i)] = float(f)
        paxAction = [paxFlow[i,j] if (i,j) in paxFlow else 0 for i,j in self.env.edges]
        return paxAction,desiredAcc
    
    def bi_level_rebalancing_2_actions(self):
        tstep = self.env.tstep
        t = self.env.time
        demandAttr = [(i,j,tt,self.env.demand[i,j][tt], self.env.price[i,j][tt]) for i,j in self.env.demand for tt in range(t,min(t+self.T, self.env.duration)) if self.env.demand[i,j][tt]>1e-3]
        accTuple = [(n,self.env.acc[n][t]) for n in self.env.acc]
        daccTuple = [(n,tt,self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t,min(t+self.T, self.env.duration))]
        edgeAttr = [(i,j,self.env.G.edges[i,j]['time']) for i,j in self.env.G.edges]
        modPath = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
        MPCPath = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/MPC/scenario_lux/bi-level-rebalancing-2-actions/'
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
           
        modfile = modPath+'bi-level-rebalancing-2-actions.mod'
        
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
        departure = defaultdict(float)
        arrival = defaultdict(float)
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
                elif item[0] == 'acc':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,f1,f2 = v.split(',')
                        f1 = float(re.sub('[^0-9e.-]','', f1))
                        f2 = float(re.sub('[^0-9e.-]','', f2))
                        departure[int(i)] = float(f1)
                        arrival[int(i)] = float(f2) 
                       
                        
        paxAction = [paxFlow[i,j] if (i,j) in paxFlow else 0 for i,j in self.env.edges]
        return paxAction,departure,arrival
        
    def tri_level(self):
        ##################
        # ADDED
        tstep = self.env.tstep
        t = self.env.time
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
            o = int(persons[persons.find('o') + 1])
            d = int(persons[persons.find('d') + 1])
            if (o, d, self.env.demand_time[o, d][t], self.env.price[o, d][t]) in flows:
                idx = flows.index((o, d, self.env.demand_time[o, d][t], self.env.price[o, d][t]))
                flow = demandAttr[idx][3] + 1
                demandAttr[idx] = (o, d, t, flow, self.env.demand_time[(o, d)][t], self.env.price[o, d][t])
            else:
                flows.append((o, d, self.env.demand_time[o, d][t], self.env.price[o, d][t]))
                demandAttr.append((o, d, t, 1, self.env.demand_time[o, d][t], self.env.price[o, d][t]))
        # Future demand
        demandAttr.extend(
            [(i, j, tt, self.env.demand[i, j][tt], self.env.demand_time[i, j][tt], self.env.price[i, j][tt]) for i, j in
             self.env.demand for tt in range(t + tstep, t + self.T, tstep) if self.env.demand[i, j][tt] > 1e-3]
        )
        #########################




        accTuple = [(n, self.env.acc[n][t + tstep]) for n in self.env.acc]
        daccTuple = [(n, tt, self.env.dacc[n][tt]) for n in self.env.acc for tt in range(t, t + self.T)]
        edgeAttr = [(i, j, self.env.reb_time[i, j][t]) for i, j in self.env.edges]
        modPath = os.getcwd().replace('\\', '/') + '/src/cplex_mod/'
        MPCPath = os.getcwd().replace('\\', '/') + '/saved_files/cplex_logs/MPC/scenario_lux/tri-level/'
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
           
        modfile = modPath+'tri-level.mod'
        
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
        desiredAcc = defaultdict(float)
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
                elif item[0] == 'desiredAcc':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,f = v.split(',')
                        f = float(re.sub('[^0-9e.-]','', f))
                        desiredAcc[int(i)] = float(f)
        return desiredAcc, paxFlow, rebFlow

