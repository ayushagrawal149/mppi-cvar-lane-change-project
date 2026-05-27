# Steps for genrating initial results

## Checking if using Highway-env is useful for generating initial results  
- Read the reports in `/home/ayush-agrawal/sq26/ece-m237/project/mppi-cvar-lane-change-project/submission/week1/week1_ECE237_AyushAgrawal_ProjectReport.pdf`, `/home/ayush-agrawal/sq26/ece-m237/project/mppi-cvar-lane-change-project/submission/week2/ECE237_AyushAgrawal_ProjectReport_Week2.pdf`, `/home/ayush-agrawal/sq26/ece-m237/project/mppi-cvar-lane-change-project/submission/week3/ECE237_AyushAgrawal_ProjectReport_Week3.pdf`  
- Read the literature in `/home/ayush-agrawal/sq26/ece-m237/project/mppi-cvar-lane-change-project/literature` 
- Then focus on the **Initial Results** section of `/home/ayush-agrawal/sq26/ece-m237/project/mppi-cvar-lane-change-project/submission/week4/ECE237_AyushAgrawal_ProjectReport_Week4.pdf`.  
- **Initial Results** section contains the high-level instructions for generating the code for building the initial results. 
- Then check the suitability of `/home/ayush-agrawal/sq26/ece-m237/project/mppi-cvar-lane-change-project/external/HighwayEnv`, whether this simulation environment can be used for creating the simulation for generating the initial results
- Check if modification to the environment (Highway-env) are extremely necessary or direct modification to the environment can be avoided with code written in separate folder (compatible in env code)
- Then prepare code base for simulating the highway lane change manoeuver for a vehicle. The vehicle dynamics is governed by bicycle model. The highway has three lanes. Add 10 vehicles for the traffic. The vehicle uses MPPI controller to successfully conduct the manoeuver. Rest details are contained in the weekly reports and literature.

## Planning intermediate step for code writing  
- Prepare the folder scaffold for the code inside the src folder.
- Write the code to setup the environment for MPPI Highway Lane Change Manoeuvre.
- Make sure the simulation runs fine 
- Write the test case such that the vehicle encounters cases such that MPPI nearly fails in certain scenario