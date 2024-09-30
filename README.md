# Learning Tethered Drone Agile Perching Strategy

This project focuses on developing an **agile** perching **strategy** for tethered drones using reinforcement learning (RL), specifically the SAC Learning from Demonstration (SACfD) technique. The RL algorithm utilized the ['Stable-Baselines3'](https://github.com/DLR-RM/stable-baselines3) library.

A complete tethered drone system (drone-tether-payload) was simulated, incorporating realistic drone dynamics, a PID controller, and a tether-payload system to model the perching process. The drone model used is a MAV model inherited from the ['gym_pybullet_drones'](https://github.com/utiasDSL/gym-pybullet-drones) project, with its compatible PID controller developed by the same team. The simulated MAV has an approximate 1:10 mass ratio, compared to the customized drone used in real-world experiments. 

The learned perching strategy demonstrated precise control, utilizing the drone’s dynamics, such as pitch angle and tether tension, to securely wrap the tether around the branch, executing smooth and efficient perching maneuvers. This was validated through real-world experiments at the Imperial College London Aerial Robotics Lab, where a **100%** success rate on selected trajactories was achieved across 52 runs, with perching completed in under 1 second — an 18.48% improvement over previous work. The SACfD agents consistently performed well on selected trajectories in both setups (20 runs in simulation and 42 runs in experiments). The actual conducted experiments including unrecored ones are up to 150 runs.

Learning from diverse (imperfect) demonstrations is proved valuable, as the SACfD agents successfully executed perching maneuvers across all test runs on selected trajectories, demonstrating smoothness, agility, and control.

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
## Main Training Script

This script handles training, evaluation (saving the best model), and testing. The training time can be adjusted by changing the `1200000` timestep parameter to fit different training goals. For example, this project shows results after 1.2 million timesteps. 

The `--show-demo` flag controls whether to display the training GUI. It is generally not recommended as it significantly reduces the training speed. Training for 1.2M timesteps usually takes around 3-4 hours, while 120k timesteps take approximately 25-30 minutes.

```
cd gym-pybullet-drones/examples/
python main.py -t 1200000 --show-demo
```

## Results
This is an example of full perching trajectory, operated by the human operator, speeded up 10 times. We can tell that human operator has relatively longer waiting time, and more conservative perching strategy.

<img src="gym_pybullet_drones/assets/full-traj-example.gif" alt="full-traj-example" width="250"/>   


### Hardware Experiment

| **Agent**               | **Traj A**                                                                                                                                   | **Traj B**                                                                                                                                   |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| SACfD - 2 Demos         | <img src="gym_pybullet_drones/assets/demo2-epi4_normal.gif" alt="demo2-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo2-epi5_normal.gif" alt="demo2-epi5" width="250"/>                                                      |
| SACfD - 5 Demos         | <img src="gym_pybullet_drones/assets/demo5-epi4_normal.gif" alt="demo5-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo5-epi5_normal.gif" alt="demo5-epi5" width="250"/>                                                      |
| SACfD - 6 Demos         | <img src="gym_pybullet_drones/assets/demo6_epi4_normal 2.gif" alt="demo6-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo6-epi5_normal.gif" alt="demo6-epi5" width="250"/>                                                      |
| SAC - 0 Demos           | <img src="gym_pybullet_drones/assets/demo0-epi4_normal.gif" alt="demo0-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo0-epi5_normal.gif" alt="demo0-epi5" width="250"/>                                                      |

## Payload Mass Effects
A lighter payload facilitates easier execution of the wrapping maneuver. As the payload mass increases, the drone requires greater thrust, torque, or velocity to reach the designated position. This, however, often leads to overshooting the target location.

<table>
  <tr>
    <th rowspan="2">10 grams</th>
    <th rowspan="2">20 grams</th>
    <th colspan="2">30 grams</th>
  </tr>
  <tr>
    <th>Unwrapped</th>
    <th>Wrapped</th>
  </tr>
  <tr>
    <td><img src="gym_pybullet_drones/assets/demo5-epi5_slow.gif" alt="10g" width="250"/></td>
    <td><img src="gym_pybullet_drones/assets/20g.gif" alt="20g" width="250"/></td>
    <td><img src="gym_pybullet_drones/assets/30_unwrap.gif" alt="demo2-epi5" width="250"/></td>
    <td><img src="gym_pybullet_drones/assets/30_wrap.gif" alt="30-wrap" width="250"/></td>
  </tr>
  <tr>
    <td>Successful - Last 1/3 Tether Contacts the Branch</td>
    <td>Too High</td>
    <td>Too High</td>
    <td>Too High & Payload Mass Balanced Drone Mass</td>
  </tr>
  <tr>
    <td>Success Rate: 3/3</td>
    <td>Success Rate: 0/3</td>
    <td colspan="2">Success Rate: 1/3</td>
  </tr>
</table>



### Simulation Testing

| **Agent**               | **Traj A**                                                                                                                                   | **Traj B**                                                                                                                                   |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| SACfD - 2 Demos         | <img src="gym_pybullet_drones/assets/demo2-epi4.gif" alt="demo2-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo2-epi5.gif" alt="demo2-epi5" width="250"/>                                                      |
| SACfD - 5 Demos         | <img src="gym_pybullet_drones/assets/demo5-epi4.gif" alt="demo5-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo5-epi5.gif" alt="demo5-epi5" width="250"/>                                                      |
| SACfD - 6 Demos         | <img src="gym_pybullet_drones/assets/demo6-epi4.gif" alt="demo6-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo6-epi5.gif" alt="demo6-epi5" width="250"/>                                                      |
| SAC - 0 Demos           | <img src="gym_pybullet_drones/assets/demo0-epi4.gif" alt="demo0-epi4" width="250"/>                                                      | <img src="gym_pybullet_drones/assets/demo0-epi5.gif" alt="demo0-epi5" width="250"/>                                                      |

### Most Effective Strategy

| **Normal Speed**                                                                                                                             | **Slow Motion**                                                                                                                             | **Corresponding Simulation**                                                                                                                        |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="gym_pybullet_drones/assets/demo6_epi4_normal 2.gif" alt="demo6-epi4-normal" width="300"/>  | <img src="gym_pybullet_drones/assets/demo6_epi4_slow-crop.gif" alt="demo6-epi4-slow" width="400"/>                                     | <img src="gym_pybullet_drones/assets/demo6-epi4.gif" alt="demo6-epi4" width="300"/>                                                                                      |
                                             
## Conclusion

The SACfD agent outperformed the SAC agent, demonstrating more deliberate control in the perching maneuvers. SAC was more aggressive but less precise. Additionally, the SAC agent managed to learn behaviors previously only achieved by SACfD in the previous work, further highlighting the improvements in the simulation environment and training system. The optimal average perching speed found in the 52 runs of experiment is betwee, 1.48 and 1.76 m/s.

The inclusion of suboptimal demonstrations played a crucial role in enhancing SACfD's adaptability, teaching the agent nuanced control techniques. Increasing the number of demonstrations is likely to further improve performance, enabling the agent to handle a wider range of scenarios. These findings underscore the effectiveness of SACfD in producing effective and agile control strategies for complex tasks like tethered drone perching.

## Future Work
- [ ] Implement dynamic training environments, including moving branches and variable tether lengths, to improve agent robustness.
- [ ] Investigate higher-level control strategies, such as velocity-based control, to enhance precision and performance beyond position control.
- [ ] Explore frameworks that directly integrate PyBullet with ROS2 for seamless simulation-to-reality transfer.
- [ ] Incorporate real-world physics elements, like wind and environmental disturbances, into the simulation to enhance realism and robustness.

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
- Tommy Woodley (2024) [*Agile Trajectory Generation for Tensile Perching with Aerial Robots*](https://github.com/TommyWoodley/TommyWoodleyMEngProject)
