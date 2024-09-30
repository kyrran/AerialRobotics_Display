# Learning Tethered Drone Agile Perching Strategy

This project focuses on developing an *agile* perching *strategy* for tethered drones using reinforcement learning (RL), specifically the SAC Learning from Demonstration (SACfD) technique. The RL algorithm utilized the ['Stable-Baselines3'](https://github.com/DLR-RM/stable-baselines3) library.

A complete tethered drone system (drone-tether-payload) was simulated, incorporating realistic drone dynamics, a PID controller, and a tether-payload system to model the perching process. The drone model used is a MAV model inherited from the ['gym_pybullet_drones'](https://github.com/utiasDSL/gym-pybullet-drones) project, with its compatible PID controller developed by the same team. The simulated MAV has an approximate 1:10 mass ratio, compared to the customized drone used in real-world experiments. 

The learned perching strategy demonstrated precise control, utilizing the drone’s dynamics, such as pitch angle and tether tension, to securely wrap the tether around the branch, executing smooth and efficient perching maneuvers. This was validated through real-world experiments at the Imperial College London Aerial Robotics Lab, where a 100% success rate was achieved across 52 runs, with perching completed in under 1 second — an 18.48% improvement over previous work. Despite some discrepancies between the simulated and real-world environments, primarily due to the 10 to $10^4$ times mass difference among the drone, tether, and payload between simulation and the experiment, the SACfD agents consistently performed well on selected trajectories in both setups (20 runs in simulation and 42 runs in experiments). 

Learning from diverse (imperfect) demonstrations proved valuable, as the SACfD agents successfully executed perching maneuvers across all test runs on selected trajectories, demonstrating smoothness, agility, and control.

The findings emphasize the importance of accurate simulation environments for transferring RL-trained behaviors to real-world applications.

## Installation

Tested on Ubuntu 22.04

```
git clone https://github.com/kyrran/gym-pybullet-drones
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip3 install --upgrade pip
pip3 install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

```

## Use

### Main Training Script

This script handles training, evaluation (saving the best model), and testing. The training time can be adjusted by changing the `1200000` timestep parameter to fit different training goals. For example, this project shows results after 1.2 million timesteps. 

The `--show-demo` flag controls whether to display the training GUI. It is generally not recommended as it significantly reduces the training speed. Training for 1.2M timesteps usually takes around 3-4 hours, while 120k timesteps take approximately 25-30 minutes.

```
cd gym-pybullet-drones/examples/
python main.py -t 1200000 --show-demo
```

## Results

### Hardware Experiment
<img src="gym_pybullet_drones/assets/full-traj-example.gif" alt="full-traj-example" width="250"/>   


### Simulation Testing

| **Agent**               | **Traj A**                                                                                                                                   | **Traj B**                                                                                                                                   |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| SACfD - 2 Demos         | <img src="gym_pybullet_drones/assets/demo2-epi4.gif" alt="demo2-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo2-epi5.gif" alt="demo2-epi5" width="250"/>                                                      |
| SACfD - 5 Demos         | <img src="gym_pybullet_drones/assets/demo5-epi4.gif" alt="demo5-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo5-epi5.gif" alt="demo5-epi5" width="250"/>                                                      |
| SACfD - 6 Demos         | <img src="gym_pybullet_drones/assets/demo6-epi4.gif" alt="demo6-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo6-epi5.gif" alt="demo6-epi5" width="250"/>                                                      |
| SAC - 0 Demos           | <img src="gym_pybullet_drones/assets/demo0-epi4.gif" alt="demo0-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo0-epi5.gif" alt="demo0-epi5" width="250"/>                                                      |


The main reasons for failed wrapping attempts in the simulation testing were either due to slow velocity, the payload losing momentum during wrapping, or the payload hitting the tether during wrapping.

The full 5 runs of the simulation testing can be referred to the image below:

<img src="gym_pybullet_drones/assets/simulation-test-img.png" alt="simulation test" width ='600'> 

Overall, the success rate of the SACfD agent in simulation is 60%, while the success rate in real-world experiments is 100%. The discrepancy between simulation and experiment is reasonable due to the extra lightweight design used in the simulation. If the exact same trajectory were executed in real-world experiments, the simulated failed trajectories during testing would likely be a successful strategy.

### Most Effective Strategy

| **Normal Speed**                                                                                                                             | **Slow Motion**                                                                                                                             | **Corresponding Simulation**                                                                                                                        |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="gym_pybullet_drones/assets/demo6_epi4_normal.gif" alt="demo6-epi4-normal" width="500"/>  | <img src="gym_pybullet_drones/assets/demo6_epi4_slow-crop.gif" alt="demo6-epi4-slow" width="400"/>                                     | <img src="gym_pybullet_drones/assets/demo6-epi4.gif" alt="demo6-epi4" width="300"/>                                                                                      |
                                             
## conclusion


 sacfd greater than sac

sacfd surpass the previous work sacfd , more smooth and agile, 

sac learn the previous sacfd better simulatio better result

suboptimal demo help, and more demo number may further facilitiate

## Future Work
- [ ] Add dynamic training environment, such as moving branch, varying tether length
- [ ] Investigate higher-level control, such as velocity control, instead of the position control used in this project
- [ ] Explore frameware that directly connects PyBullet and ROS2
- [ ] Integrate real-world physics like wind into simulation environement

## Troubleshooting

- [Official Method] On Ubuntu, with an NVIDIA card, if you receive a "Failed to create and OpenGL context" message, launch `nvidia-settings` and under "PRIME Profiles" select "NVIDIA (Performance Mode)", reboot and try again.

Run all tests from the top folder with

```
pytest tests/
```
- If the method above doesn't fix it, and conda / miniconda is used, please edit .bashrc
```
export MESA_GL_VERSION_OVERRIDE=3.2
export MESA_GLSL_VERSION_OVERRIDE=150
```

- Sometimes, this method can also work:
```
conda install -c conda-forge libgcc=5.2.0
conda install -c anaconda libstdcxx-ng
conda install -c conda-forge gcc=12.1.0
```


## References

- Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig (2021) [*Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control*](https://arxiv.org/abs/2103.02142) 
- Antonin Raffin, Ashley Hill, Maximilian Ernestus, Adam Gleave, Anssi Kanervisto, and Noah Dormann (2019) [*Stable Baselines3*](https://github.com/DLR-RM/stable-baselines3)
- F. Hauf et al., [*Learning Tethered Perching for Aerial Robots*](https://ieeexplore.ieee.org/document/10161135) 2023 IEEE International Conference on Robotics and Automation (ICRA), London, United Kingdom, 2023, pp. 1298-1304, doi: 10.1109/ICRA48891.2023.10161135. 
