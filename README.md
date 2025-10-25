# FRE25 IsaacLab Simulation
![Demonstartion](https://previews.dropbox.com/p/thumb/ACwIycsJZz19KxGXPcZIgOQrZYW9sdVxWIJbGoEs-lVDPXmfKwz9wrLLzt1rpQei7FJ40bPRAGIfmuqHgvmjAmz_mYSwvqzS-_xiBBZjIcJ3vp6OZw_bq9QHNQF-MDm8HTPvwa9nzhME6iVZ1i8nb8xZ4ivJI6v_1fmjfKSgYz9v7DxKTxECwX5phRC0840rXngvBsGqFBcCqhHW2i9beXJCp3NqnGuwbe1dRYriNv2HolyGmliU4Nn14suiyQKjkU3O68flLWJiOCxVlFd1-AyOyGE2KsAnouletrx0hVz_m_A54LxLvsYgrE330TnI-uakCVif-iPIW1GOlbF_QvYV3Y37o2vKFnmRzFAVr2g_bq2p5PfzB7Pu7RnkYYrzxqrBUVtqo9FZbqL0GZlCjQIk51AteUQyfmAWvnP76oE31hf5qHHlBLJSSAseCnxvxTb_4FooTNZuCR5fXWzLkRrAuGUoSmfbt1C3EouP-yleWHTYp3fdfK2COtCLS6nDgfC4VJUKzBO8gjVcH_xdy1iwjQhnLQkKWpR20nbTfasL1B_8sL6q5cvVILTGo7e_jSruNDabJVOVTroWOdu2_mp8FXtodj6VpLA3ilkyMXgP5yTkOPRKwvGaNSPSnujV6F6juHsIZN2zfLDDD6H-GLAoTjcS3rwdt8Axq1JxWbzEz5N36GRvXG2hN5m9qzH97KvGq9rOIX9321_RYj5IN5m0dVhNBqhFO8HOVqdnj0gWolZhBzrlbfgbAbISUCXqqty-JWr315HuOQvGX_fUNrSYupYQEdpzdbhBM55wE87dZRxlRGnLywzruNYZvd0Jb3OVb4JP-EokimLHAEcigs6uHJclwuhrOZYS2jJ5kwXKYJRCiFIyMNWsk3dEf1Ttt46BXVXbO_4oXmJV8MoasH8VzGx2u_pIpcCvk5cmHnzIYfYBBDFCufLp60jgMUjfvdg-RIe0E9YLSicwBKXZKW5_WsbUSJdyxmiV9CjFP5IGjJfzqxXXfTV972xrW_uO6I_t7YDHl1cbvAyfljsEvvIUy7Mp2BHQuZUJg4yFg9MqX8EPHimvnqEZZNIiVCIExd6KD_a1q2BpUzOqVTkyaAXDcNARiMWTyhhFcWbL06PYtHDNUA3m3fJO_ebhIqdHUCmi4mzNaLNqH0WXlrIpHGOT3OeU2uJYAby0W4Jssf_d9A/p.gif?is_prewarmed=true)
## Overview

This project explores end-to-end Reinforcement Learning for autonomous agricultural navigation, specifically targeting Task 1 of the Field Robot Event 2025 (FRE25). Using Politecnico di Milano's Rockerbot platform with its unique 4WIS4WID (4 Wheels Independent Steering 4 Wheels Independent Drive) kinematic configuration, the project implements a high-fidelity GPU-accelerated simulation environment in NVIDIA Isaac Lab.

**Project Goal:** Develop an RL policy that navigates crop rows by directly processing LIDAR sensor data and outputting control commands, without relying on traditional Sense-Plan-Act pipelines.

**Current Status:** Completed simulation development and curriculum level 4 training (Crab kinematic with command buffer control). The project successfully trains policies that navigate crop rows within ~2 minutes, achieving full row completion in ~1 hour of training.

**Keywords:** reinforcement learning, agricultural robotics, Isaac Lab, autonomous navigation, 4WIS4WID, PPO, sim-to-real

---

## Key Features

### Simulation Environment
- **GPU-Accelerated Physics:** Achieves up to 3000 simulation steps/second using Isaac Lab's PhysX backend
- **Massively Parallel Training:** Supports 4096+ environments running simultaneously
- **Custom Ray Marching LIDAR:** From-scratch implementation for dynamic plant detection using signed distance fields
- **Procedural Scene Generation:** Randomized crop row layouts with configurable curvature, spacing, and plant density
- **High-Fidelity Robot Model:** Full Rockerbot CAD model with independent wheel steering and drive control

### Learning Framework
- **Discrete Action Space:** Simplified differential control (steering, throttle, command buffer)
- **Accumulator-Based Memory:** Novel memory architecture avoiding RNN/LSTM computational overhead
- **Curriculum Learning:** Progressive difficulty scaling from simple navigation to complex multi-row tasks
- **Comprehensive Reward Shaping:** Multi-component reward balancing waypoint reaching, velocity, and constraint satisfaction

### Supported RL Libraries
- **Stable Baselines 3 (SB3):** Primary training framework with PPO implementation
- **SKRL:** Alternative training backend with multi-algorithm support
- Extensible architecture for additional RL frameworks

---

## Quick Start

### Prerequisites
- Isaac Lab installation (see [Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html))
- Python 3.10+
- NVIDIA GPU with CUDA support
- Recommended: Conda environment for isolated dependency management

### Installation

1. **Install Isaac Lab** following the official guide. Conda installation is recommended.

2. **Clone this repository** outside the Isaac Lab directory:
```bash
git clone https://github.com/AIRLab-POLIMI/FRE25_IsaacLabSym.git
cd FRE25_IsaacLabSym
```

3. **Install the extension** in editable mode:
```bash
# Use isaaclab.sh -p instead of python if Isaac Lab not in venv/conda
python -m pip install -e source/FRE25_IsaacLabSym
```

4. **Verify installation** by listing available environments:
```bash
python scripts/list_envs.py
```

Expected output should include `Fre25-Isaaclabsym-Direct-v0`.

---

## Usage

### Training

#### Using Shell Scripts (Recommended)
The repository provides convenient shell scripts for common operations:

```bash
# Train with Stable Baselines 3 (128 environments, headless mode)
./RUN_SB3_TRAIN.sh

# Train with custom number of environments
./RUN_SB3_TRAIN.sh --num_envs 512

# Train with Hydra configuration override
./RUN_SB3_TRAIN_HYDRA.sh
```

#### Direct Training Commands

```bash
# SB3 training
ISAAC_LAB_PATH="/path/to/IsaacLab"
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/sb3/train.py \
    --task Fre25-Isaaclabsym-Direct-v0 \
    --num_envs 128 \
    --headless

# SKRL training with Hydra config
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/skrl/train.py \
    --task Fre25-Isaaclabsym-Direct-v0 \
    --num_envs 128 \
    --headless
```

### Evaluation

```bash
# Play trained policy
./RUN_SB3_PLAY.sh

# Or specify checkpoint manually
${ISAAC_LAB_PATH}/isaaclab.sh -p scripts/sb3/play.py \
    --task Fre25-Isaaclabsym-Direct-v0 \
    --num_envs 1 \
    --checkpoint /path/to/model.zip
```

### Teleoperation

Test the environment with keyboard control:

```bash
./RUN_TELEOP.sh
```

**Controls:**
- `W/S`: Throttle forward/backward
- `A/D`: Steer left/right
- `E` (hold): Step command buffer
- `L`: Reset controller state

See [KEYBOARD_CONTROLS.md](KEYBOARD_CONTROLS.md) for detailed control documentation.

### Monitoring Training

```bash
# Launch TensorBoard
./RUN_TENSORBOARD.sh

# Or manually specify log directory
tensorboard --logdir logs/sb3/Fre25-Isaaclabsym-Direct-v0
```

### Testing with Dummy Agents

```bash
# Random agent
python scripts/random_agent.py --task Fre25-Isaaclabsym-Direct-v0

# Zero-action agent
python scripts/zero_agent.py --task Fre25-Isaaclabsym-Direct-v0
```

---

## Configuration

All simulation and training parameters are centralized in `source/FRE25_IsaacLabSym/FRE25_IsaacLabSym/tasks/direct/fre25_isaaclabsym/fre25_isaaclabsym_env_cfg.py`.

### Key Parameters

#### Environment Setup
- `episode_length_s = 300.0`: Episode duration in seconds
- `decimation = 4`: Physics steps per control step
- `num_envs`: Number of parallel environments (set at runtime)

#### Robot Control
- `steering_scale = 2`: Steering angle change per action step [deg/step]
- `wheels_effort_scale = 15`: Wheel motor velocity scale
- `steering_buffer_min/max = ±π/2`: Steering angle limits [rad]

#### Scene Generation
- `path_length = 3.0`: Row length [m]
- `paths_spacing = 1.2`: Row spacing (Δᵣ) [m]
- `n_control_points = 10`: Control points for spline generation (cᵣ)
- `n_plants_per_path = 10`: Plants per row (ρ)
- `plant_radius = 0.22`: Plant collision radius (rₚₗₐₙₜ) [m]

#### LIDAR Sensor
- `lidar_rays_per_robot = 40`: Number of rays (nᵣₐᵧₛ)
- `lidar_max_distance = 1.0`: Max sensing range [m]
- `lidar_tolerance = 0.01`: Ray marching convergence tolerance (εₗᵢ��ₐᵣ) [m]
- `lidar_max_steps = 100`: Max ray marching iterations (nₗᵢ��ₐᵣ)

#### Waypoint System
- `waypoint_reached_epsilon = 0.35`: Reached threshold (εw) [m]
- `waypoints_per_row = 3`: Waypoints per row

#### Reward Function
- `waypoint_reward_base = 100.0`: Waypoint completion reward
- `velocity_towards_waypoint_scale = 0.5`: Velocity projection reward weight
- `plant_collision_penalty = -100.0`: Plant collision penalty
- `total_reward_scale = 0.1`: Final reward scaling factor

For complete parameter documentation with units and descriptions, see the configuration file.

---

## Project Structure

```
FRE25_IsaacLabSym/
├── scripts/
│   ├── sb3/                          # Stable Baselines 3 training/play scripts
│   │   ├── train.py                  # SB3 training entry point
│   │   ├── play.py                   # Policy evaluation
│   │   └── callbacks.py              # Custom logging callbacks
│   ├── skrl/                         # SKRL training/play scripts
│   ├── teleop/                       # Keyboard teleoperation
│   ├── random_agent.py               # Random action agent
│   └── zero_agent.py                 # Zero action agent
│
├── source/FRE25_IsaacLabSym/
│   └── FRE25_IsaacLabSym/
│       └── tasks/
│           ├── direct/               # Direct RL environment (main)
│           │   └── fre25_isaaclabsym/
│           │       ├── fre25_isaaclabsym_env.py         # Environment implementation
│           │       ├── fre25_isaaclabsym_env_cfg.py     # Configuration
│           │       ├── RockerBot.py                     # Robot model
│           │       ├── CommandBuffer/                   # Command buffer logic
│           │       ├── PathHandler.py                   # Row generation
│           │       ├── PlantRelated/                    # Plant spawning/collision
│           │       ├── RayMarcher/                      # Custom LIDAR implementation
│           │       ├── WaypointRelated/                 # Waypoint system
│           │       ├── Assets/                          # 3D models (robot, plants)
│           │       └── agents/                          # RL agent configs
│           │
│           └── manager_based/        # Manager-based environment (legacy)
│
├── logs/                             # Training logs
│   ├── sb3/                          # SB3 logs and checkpoints
│   └── skrl/                         # SKRL logs and checkpoints
│
├── outputs/                          # Hydra experiment outputs
│
├── Multidisciplinary_Project_Report/ # Academic report (LaTeX)
│   ├── executive_summary.tex         # Main report document
│   └── Images/                       # Report figures
│
├── ExperimentalNotebooks/            # Jupyter notebooks for analysis
│
├── RUN_*.sh                          # Convenience shell scripts
├── KEYBOARD_CONTROLS.md              # Teleoperation documentation
└── README.md                         # This file
```

---

## Curriculum Learning Progression

The project follows a curriculum to progressively increase task difficulty:

1. **Level 5 (Current):** Plain navigation - Crab kinematics, automatic command buffer, skip 1 row
2. **Level 4:** Command buffer control - Crab kinematics, manual buffer control, skip 1 row ✓
3. **Level 3:** Multi-row skipping - Crab kinematics, manual buffer, skip up to 2 rows
4. **Level 2:** Full rigid body - FRBK kinematics with rotation, manual buffer
5. **Level 1 (Goal):** Full 4WIS4WID - Independent wheel control, complete task specification

**Progress:** Levels 5-4 completed. Level 4 trains to first row completion in ~2 minutes and full competence in ~1 hour.

---

## Training Results

After 339 training runs (487.29 hours total experimentation):

- **Simulation Performance:** Up to 3000 steps/second on GPU
- **Training Time:** ~1 hour from scratch to competent navigation
- **First Success:** First row completion in ~2 minutes of training
- **Configuration:** 4096 parallel environments, PPO with discrete actions
- **Observation Space:** 47 dimensions (steering, LIDAR, command buffer, past actions)
- **Action Space:** 3 discrete actions (steering, throttle, command step)

Key findings documented in `Multidisciplinary_Project_Report/executive_summary.tex`.

---

## Advanced Features

### Observation Space Composition
- Current steering angle: 1D
- LIDAR readings: 40D (configurable via `lidar_rays_per_robot`)
- Command buffer state: 3D (current command, parity, buffer state)
- Past actions: 3D (previous steering, throttle, command step)
- **Total:** 47D base observation space

### Action Space Design
Discrete MultiDiscrete space for stability and sample efficiency:
- **Steering:** {-1, 0, +1} (left, neutral, right)
- **Throttle:** {-1, 0, +1} (backward, stop, forward)
- **Command Step:** {0, 1} (hold, advance buffer)
- **Hidden States (optional):** {-1, +1}ⁿ for memory augmentation

### Memory Architecture
Novel accumulator-based approach avoiding RNN computational overhead:
- Differential hidden state control: $\dot{\vec{h}} \in \{-1, +1\}^{n_h}$
- Continuous integration: $\vec{h}_{t+1} = \vec{h}_t + \alpha \cdot \dot{\vec{h}}$
- Provides policy with "scratchpad" for state tracking
- Fully parallel implementation compatible with massive batching

---

## Development Tools

### VSCode IDE Setup
Run VSCode task `setup_python_env` (Ctrl+Shift+P → Tasks: Run Task) to configure Python paths for Omniverse extensions.

### Code Formatting
```bash
pip install pre-commit
pre-commit run --all-files
```

### Omniverse Extension (Optional)
Enable UI extension in Isaac Sim:
1. Add `source/` directory to Extensions search paths
2. Enable "FRE25_IsaacLabSym" under Third Party extensions

---

## Troubleshooting

### Pylance Indexing Issues
Add extension path to `.vscode/settings.json`:
```json
{
    "python.analysis.extraPaths": [
        "/path/to/FRE25_IsaacLabSym/source/FRE25_IsaacLabSym"
    ]
}
```

### Memory Issues with Pylance
Exclude unused Omniverse packages in `.vscode/settings.json` under `python.analysis.extraPaths`.

### GPU Memory Errors
Reduce `num_envs` or adjust PhysX GPU memory settings in `fre25_isaaclabsym_env_cfg.py`:
```python
gpu_max_rigid_contact_count=2**23
gpu_collision_stack_size=2**28
```

### LIDAR Performance
Raymarching parameters can be tuned for performance vs accuracy:
- Reduce `lidar_rays_per_robot` for faster computation
- Adjust `lidar_max_steps` for raymarching precision

---

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{ginefra2025fre25,
    title={4WIS4WID Mobile Robot Autonomous Navigation in Agricultural Setting using End-to-End Reinforcement Learning},
    author={Ginefra, Paolo},
    year={2025},
    school={Politecnico di Milano},
    type={Multidisciplinary Project},
    note={Supervisors: M. Restelli, S. Mentasti, M. Matteucci}
}
```

---

## Related Resources

- **Isaac Lab Documentation:** [isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab/)
- **Field Robot Event:** [fieldrobot.com](https://www.fieldrobot.com/)
- **Rockerbot Platform:** Politecnico di Milano AIRLab
- **GitHub Repository:** [github.com/AIRLab-POLIMI/FRE25_IsaacLabSym](https://github.com/AIRLab-POLIMI/FRE25_IsaacLabSym)

---

## Future Work

Planned extensions and research directions:

1. **Sim-to-Real Transfer:** Domain randomization, sensor noise modeling, real Rockerbot deployment
2. **Curriculum Progression:** Advance to full 4WIS4WID control (Level 1)
3. **Ablation Studies:** Systematic evaluation of memory architecture, action discretization, reward components
4. **Generalization Testing:** Evaluation on unseen row curvatures, plant densities, spacing variations
5. **Alternative Approaches:** Comparison with imitation learning, classical controllers, other RL algorithms

See `Multidisciplinary_Project_Report/executive_summary.tex` Section 6 for detailed future work proposals.

---

## License

This project is licensed under the BSD-3-Clause License, consistent with Isaac Lab licensing.

---

## Acknowledgments

- **NVIDIA Isaac Lab Team** for the simulation framework
- **Politecnico di Milano AIRLab** for Rockerbot platform and support
- **Field Robot Event Organization** for the challenge specification
- **Supervisors:** Prof. M. Restelli, Prof. S. Mentasti, Prof. M. Matteucci

---

## Contact

**Author:** Paolo Ginefra  
**Institution:** Politecnico di Milano - School of Industrial and Information Engineering  
**Academic Year:** 2024-2025

For questions or collaboration inquiries, please open an issue on the GitHub repository.
