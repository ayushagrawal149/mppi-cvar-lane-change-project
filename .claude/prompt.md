# Role:  
You are acting as a Senior Control SYstems Engineer specializing in Stochastic Optimal Control and Reinforcement Learning

# Objective:  
Analyze the uploaded research papers in `/home/ayush-agrawal/sq26/ece-m237/project/mppi-cvar-lane-change-project/literature` to extract the core components required to replicate the Problem Modelling and the Technical Sketch of Work. 


# Task 1: Problem Modelling Extraction:    
For each distinct methodology presented (e.g., CVaR-MDP, MPPI, RA-MPPI), identify and define:  

- **The State Space (X)**: Define the physical states (e.g., position, velocity, orientation) and any Augmented States (e.g., confidence level $\alpha$, running cost y, or auxilliary variables $s$).

- **Dynamics and Stochasticity**: Transcribe the transfer function $F(x, u)$ or the Stochastic Differential Equation (SDE). Specifically, note if the noise if additive, multiplicative, or follows a specific distribution (e.g., Brownian Motion)

- **The Objective Function**: Extract the mathematical definition of the cost/reward. Distinguish between the **Stage-wise Cost** $q(x,u)$ and the *Risk Measure* (e.g., CVaR, Mean-CVaR, or KL-Divergence)

- **Constraints**: Identify any hard constraints or "Soft Trajectory Filtering" mechanisms mentioned.


# Task 2: Sketch of Work (Algorithmic Logic)  
Provide a high-level technical sketch of the solution approach. Include:  

- **The Recursive Step**: Trancribe the specific Bellman Operator, Policy Gradient formula, or Path Integral update law used.

- **The Approximation Method**: How does the paper handle continuous spaces? (e.g., Linear Interpolation, Monte-Carlo sampling on GPUs, or Linear Function Approximation)

- **Verification Logic**: Note any "Self-Audit" or "Safety-Aware" loops described in the papers to prevent collisions or catastrophic failures.

# Formatting Requirements:
1. Use **LaTeX** for all mathematical expressions.
2. Present defintions in a **Variable Mapping Table**.
3. Provide the Sketch of Work as a **Numbered Technical Summary**.