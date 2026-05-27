# mppi-cvar-lane-change-project
This is the code base for the course project on ECEM237 - Dynamic Programming. The project is developed by Ayush Agrawal with assistance from Claude Opus 4.7.


## Reproducing the environment

### One-time setup
\`\`\`bash
git clone --recurse-submodules https://github.com/<you>/mppi-cvar-lane-change-project.git
cd mppi-cvar-lane-change-project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e external/HighwayEnv
\`\`\`

### If you forgot --recurse-submodules
\`\`\`bash
git submodule update --init --recursive
\`\`\`

### Running an experiment
\`\`\`bash
python -m src.experiments.run_baseline
\`\`\`

# Setup and run  
Following are the commands need to be run to setup the repository with the Highway-Env simulation environment  
> git clone --recurse-submodules <your-repo-url>  
> cd mppi-cvar-lane-change-project  
> python3 -m venv .venv && source .venv/bin/activate  
> pip install -r requirements.txt && pip install -e external/HighwayEnv  
> python -c "import highway_env, gymnasium as gym; gym.make('highway-v0').reset  (); print('ready')" 
