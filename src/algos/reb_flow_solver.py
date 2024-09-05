import os
import subprocess
from collections import defaultdict
from src.misc.utils import mat2str
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, value

def solveRebFlow(env,res_path,desiredAcc,CPLEXPATH, directory):
    t = env.time
    accRLTuple = [(n,int(round(desiredAcc[n]))) for n in desiredAcc]
    accTuple = [(n,int(env.acc[n][t+1])) for n in env.acc]
    edgeAttr = [(i,j,env.G.edges[i,j]['time']) for i,j in env.G.edges]
    modPath = os.getcwd().replace('\\','/')+'/src/cplex_mod/'
    OPTPath = os.getcwd().replace('\\','/')+'/' + directory +'/cplex_logs/rebalancing/'+res_path + '/'
    if not os.path.exists(OPTPath):
        os.makedirs(OPTPath)
    datafile = OPTPath + f'data_{t}.dat'
    resfile = OPTPath + f'res_{t}.dat'
    with open(datafile,'w') as file:
        file.write('path="'+resfile+'";\r\n')
        file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
        file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
        file.write('accRLTuple='+mat2str(accRLTuple)+';\r\n')
    modfile = modPath+'minRebDistRebOnly.mod'
    if CPLEXPATH is None:
        CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file =  OPTPath + f'out_{t}.dat'
    with open(out_file,'w') as output_f:
        subprocess.check_call([CPLEXPATH+"oplrun", modfile, datafile], stdout=output_f, env=my_env)
    output_f.close()

    # 3. collect results from file
    flow = defaultdict(float)
    with open(resfile,'r', encoding="utf8") as file:
        for row in file:
            item = row.strip().strip(';').split('=')
            if item[0] == 'flow':
                values = item[1].strip(')]').strip('[(').split(')(')
                for v in values:
                    if len(v) == 0:
                        continue
                    i,j,f = v.split(',')
                    flow[int(i),int(j)] = float(f)
    action = [flow[i,j] for i,j in env.edges]
    return action

def solveRebFlow_pulp(env, desiredAcc):

    t = env.time
    
    # Prepare the data: rounding desiredAcc and getting current vehicle counts
    accRLTuple = [(n, int(round(desiredAcc[n]))) for n in desiredAcc]
    accTuple = [(n, int(env.acc[n][t+1])) for n in env.acc]
    
    # Extract the edges and the times
    edgeAttr = [(i, j, env.G.edges[i, j]['time']) for i, j in env.G.edges]
    edges = [(i, j) for i, j, _ in edgeAttr]

    # Map vehicle availability and desired vehicles for each region
    acc_init = {n: env.acc[n][t+1] for n in env.acc}
    desired_vehicles = {n: int(round(desiredAcc[n])) for n in desiredAcc}

    region = [n for n in env.acc]
    # Time on each edge (used in the objective)
    time = {(i, j): env.G.edges[i, j]['time'] for i, j in env.G.edges}

    # Define the PuLP problem
    model = LpProblem("RebalancingFlowMinimization", LpMinimize)

    # Decision variables: rebalancing flow on each edge
    rebFlow = {(i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat='Integer') for (i, j) in edges}

    # Objective: minimize total time (cost) of rebalancing flows
    model += lpSum(rebFlow[(i, j)] * time[(i, j)] for (i, j) in edges), "TotalRebalanceCost"

    # Constraints for each region (node)
    for i in region:
        # 1. Flow conservation constraint (ensure net inflow/outflow achieves desired vehicle distribution)
        model += (
            lpSum(rebFlow[(j, i)] for (j, i) in edges if j != i) - 
            lpSum(rebFlow[(i, j)] for (i, j) in edges if i != j)
        ) >= desired_vehicles[i] - acc_init[i], f"FlowConservation_{i}"

        # 2. Rebalancing flows from region i should not exceed the available vehicles in region i
        model += (
            lpSum(rebFlow[(i, j)] for (i, j) in edges if i != j) <= acc_init[i], 
            f"RebalanceSupply_{i}"
        )
    
    # Solve the problem
    status = model.solve()

    # Check if the solution is optimal
    if LpStatus[status] == "Optimal":
        # Collect the rebalancing flows
        flow_result = {(i, j): value(rebFlow[(i, j)]) for (i, j) in edges}

        # Map the flow values to the output array based on the edges
        action = [flow_result.get((i, j), 0) for (i, j) in env.edges]
        return action
    else:
        print(f"Optimization failed with status: {LpStatus[status]}")
        return None

