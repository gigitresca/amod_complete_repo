# Data-driven Methods for Network-level Coordination of Autonomous Mobility-on-Demand Systems Across Scales
Official implementation of the [Data-driven Methods for Network-level Coordination of Autonomous Mobility-on-Demand Systems Across Scales](https://rl4amod-itsc24.github.io/) tutorial, presented at 27th IEEE International Conference on Intelligent Transportation Systems 2024

<img src="figures/gnn-for-amod.png" width="700"/></td> <br/>

## Prerequisites

* You will need to have a working IBM CPLEX installation. If you are a student or academic, IBM is releasing CPLEX Optimization Studio for free. You can find more info [here](https://community.ibm.com/community/user/datascience/blogs/xavier-nodet1/2020/07/09/cplex-free-for-students)


* You will need to have a working installation of SUMO (Simulation of Urban MObility). It is an open source microscopic and continous multi-modal traffic simulation package used to handle large networks. The repo is based on the version 1.20.0. Find the information for SUMO installation [here](https://sumo.dlr.de/docs/Installing/index.html)


* To install all required dependencies, run
```
pip install -r requirements.txt
```
## Contents
`amod_complete_repo`  
`├── figures/`: directory for saving the output figures<br>
`├── main_a2c.py`  
`├── main_mpc.py`  
`├── main_noreb.py`  
`├── main_plusone.py`  
`├── main_sac.py`  
`├── main_uniform.py`  
`├── saved_files/`: directory for saving results, logging, etc<br>
`├── src/`  
`│   ├── algos/`  
`│   │   ├── a2c.py`: PyTorch implementation of A2C-GNN<br>
`│   │   ├── base.py`: base class for baseline algorithms<br>
`│   │   ├── ed.py`: implementation of an Equal Distribution policy<br>
`│   │   ├── MPC.py`: implementation of the MPC policy<br>
`│   │   ├── no_reb.py`: implementation of a no rebalancing policy<br>
`│   │   ├── plus_one.py`: implementation of a plus1 policy<br>
`│   │   ├── random.py`: implementation of a random policy<br>
`│   │   ├── reb_flow_solver.py`: thin wrapper around CPLEX formulation of the Rebalancing problem<br>
`│   │   ├── registry.py`: models registration function<br>
`│   │   └── sac.py`: Pytorch implementation of SAC-GNN<br>
`│   ├── config/`: folder with the default parameters for the simulator and the models written in hydra<br>
`│   ├── cplex_mod/`: CPLEX formulation of Rebalancing, Matching and MPC problems<br>
`│   ├── envs/`  
`│   │   ├── data/`: folder with the data for the macroscopic and mesoscopic scenarios (SUMO input files and json files for the demand)<br>
`│   │   └── sim/`  
`│   │      ├── macro_env.py`: implementation of a macroscopic simulator for AMoD systems system<br>
`│   │      └── sumo_env.py`: implementation of a SUMO-based mesoscopic simulator for AMoD systems<br>
`│   ├── misc/`: helper functions<br>
`│   ├── nets/`  
`│   │   ├── actor.py`: Pytorch implementation of a GNN-based actor<br>
`│   │   ├── critic.py`: Pytorch implementation of a GNN-based critic<br>
`├── test.ipynb`: test main file<br>
`├── train.py`: Rl agents train main file <br>

## Examples

To train an agent, `main.py` accepts the following arguments:
```bash
cplex arguments:
    --cplexpath     defines directory of the CPLEX installation
    
model arguments:
    --test          activates agent evaluation mode (default: False)
    --max_episodes  number of episodes to train agent (default: 16k)
    --max_steps     number of steps per episode (default: T=60)
    --no-cuda       disables CUDA training (default: True, i.e. run on CPU)
    --directory     defines directory where to log files (default: saved_files)
    
simulator arguments: (unless necessary, we recommend using the provided ones)
    --seed          random seed (default: 10)
    --demand_ratio  (default: 0.5)
    --json_hr       (default: 7)
    --json_tsetp    (default: 3)
    --no-beta       (default: 0.5)
```

**Important**: Take care of specifying the correct path for your local CPLEX installation. Typical default paths based on different operating systems could be the following
```bash
Windows: "C:/Program Files/ibm/ILOG/CPLEX_Studio128/opl/bin/x64_win64/"
OSX: "/Applications/CPLEX_Studio128/opl/bin/x86-64_osx/"
Linux: "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
```
### Training and simulating an agent

1. To train an agent (with the default parameters) run the following:
```
python main.py
```

2. To evaluate a pretrained agent run the following:
```
python main.py --test=True
```

## Credits
This work was conducted as a joint effort with [Kaidi Yang*](https://sites.google.com/site/kdyang1990/), [James Harrison*](https://stanford.edu/~jh2/), [Filipe Rodrigues'](http://fprodrigues.com/), [Francisco C. Pereira'](http://camara.scripts.mit.edu/home/) and [Marco Pavone*](https://web.stanford.edu/~pavone/), at Technical University of Denmark' and Stanford University*. 

## Reference
```
@inproceedings{GammelliYangEtAl2021,
  author = {Gammelli, D. and Yang, K. and Harrison, J. and Rodrigues, F. and Pereira, F. C. and Pavone, M.},
  title = {Graph Neural Network Reinforcement Learning for Autonomous Mobility-on-Demand Systems},
  year = {2021},
  note = {Submitted},
}
```

----------
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at daga@dtu.dk.