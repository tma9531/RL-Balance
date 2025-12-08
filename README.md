# Balancing Graphics and Performance in Video Games Using Reinforcement Learning
**Author:** Tyler Alvarez  
Repository for the research project and paper  
**“Balancing Graphics and Performance in Video Games Using RL.”**

---

## Overview

This repository contains the reinforcement learning scripts and data used in the study. The project investigates whether a reinforcement learning (RL) agent can automatically tune graphics settings to achieve an effective balance between **frame rate (FPS)** and **visual quality**, especially for players with lower-end hardware.

This repository does not include the Unity environment and project due to Unity copyright rules.

Two models were developed:

1. **Real Model** — Observes real FPS directly from a Unity environment.
2. **Simulated Model** — Uses a differentiable GPU-load approximation to estimate FPS.

The agent is trained using **Proximal Policy Optimization (PPO)** with **Unity ML-Agents**.
  


---

## Training Model Summary

### Real Model (Unity FPS Observation)

- Uses the **TerrainURP** Unity demo environment.
- The agent:
  - Adjusts four graphics settings: resolution, texture quality, shadow quality, and anti-aliasing.
  - Observes real FPS measured directly from the Unity environment.
- Training is limited to **~4 steps per second**:
  - FPS needs time to stabilize after each settings change.
  - Faster step rates produce noisy, unreliable FPS readings.
- Advantages:
  - High **fidelity** to the actual game environment.
- Drawbacks:
  - **Very slow** training.
  - Reward and convergence curves are **noisy** due to FPS jitter, OS scheduling, and GPU behavior.

### Simulated Model (Approximate FPS Function)

- Uses a handcrafted GPU-load model instead of real FPS:

  \[
  L = 2.5r^2 + 2.0s^2 + 0.6t + 1.2\sqrt{a}
  \]

  where:
  - \( r \): resolution level  
  - \( t \): texture quality  
  - \( s \): shadow quality  
  - \( a \): anti-aliasing level  

- Simulated FPS is computed as:

  \[
  FPS_{\text{sim}} = \frac{FPS_{\text{target}}}{L + 1}
  \]

- A small random noise term (e.g., ±2%) is added to mimic real system variability.
- Advantages:
  - **Very fast** training (limited only by CPU speed).
  - Ideal for **reward shaping**, ablations, and hyperparameter exploration.
- Drawbacks:
  - Only an **approximation** of real GPU behavior.
  - Policies trained purely in simulation transfer with **lower-quality** graphics when deployed in the real Unity environment.

### Common RL Setup

- Algorithm: **Proximal Policy Optimization (PPO)** via Unity ML-Agents.
- Observations:
  - Current graphics settings.
  - Current FPS (real or simulated).
  - Previous FPS (for FPS delta term).
- Actions:
  - Change one graphics setting up or down (discrete) in this implementation.
- Reward:
  - Balances FPS level, FPS improvement, and visual quality using:

    \[
    R = \left( \frac{FPS}{FPS_{\text{target}}} \right)^2
      + \beta(FPS_t - FPS_{t-1})
      - \alpha\left(1 - \sqrt{0.4r + 0.3t + 0.2s + 0.1a}\right)
    \]

- Goal:
  - Learn a policy that finds a **practical tradeoff** between frame rate and graphics quality for lower-end hardware.
