# HAL-9000

## Overview
HAL-9000 is a collection of AI autopilots trained using reinforcement learning for integration with the Outer Wilds Web project. These AIs are designed to assist with autonomous navigation and decision-making within the game environment. The project is developed by Quentin Rollet, Marc-Antoine Vergnet, and Patrice Soulier, who also lead the development of Outer Wilds Web.

## Requirements

Ensure you have the following dependencies installed. While other versions might work, they have not been tested:

* **Python:** 3.13.2
* **npm:** 11.1.0
* **Cargo (Rust):** 1.85.0

## Quick Start Guide

Follow these steps to set up the environment and run the HAL-9000 agents.

### 1. Setup Python Environment (HAL-9000 Core)

This sets up the environment for the AI training and execution scripts.

**Create and activate a virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
# On Windows use: .venv\Scripts\activate
```

**Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**Configure environment variables:**

Create a file named `.env` in the root directory of the HAL-9000 project and add the following variables:

```bash
echo "SLEEP_TIME=0.015" >> .env
echo "NUMBER_EPISODE=15000" >> .env
echo "EPISODE_TIME=15" >> .env
echo "SAVE_NUMBER=100" >> .env
```

**Variable Explanations:**

* `SLEEP_TIME`: Pause duration (in seconds) between steps executed by the Python environment. This controls how fast the Python script interacts with the simulation. A value of `0.25` approximates real-time interaction speed. A smaller value like `0.015` results in much faster execution (~40 steps per second, assuming the server can keep up).
* `NUMBER_EPISODE`: The total number of episodes to run during a training session.
* `EPISODE_TIME`: Maximum duration of a single training episode in *simulated game minutes*. The actual wall-clock time this takes depends on the simulation speed set in the Rust server and the `SLEEP_TIME` in this Python environment.
* `SAVE_NUMBER`: Frequency for saving the trained model (e.g., a value of `100` saves the model every 100 episodes).

### 2. Set Up the Rust Server (Outer Wilds Web Simulation)

HAL-9000 interacts with the Rust-based server that runs the Outer Wilds Web simulation.

**Clone and prepare the server repository:**

```bash
git clone https://github.com/outer-wilds-web/rust-server.git
cd rust-server
git checkout deep_learning # Switch to the required branch
```

**Configure server environment variables:**

Create a file named `.env` inside the `rust-server` directory:

```bash
echo "SIMULATION_SLEEP_TIME_MICROSECONDS=1600" >> .env
echo "SERVER_SLEEP_TIME_MICROSECONDS=6400" >> .env
```

**Variable Explanations:**

* `SIMULATION_SLEEP_TIME_MICROSECONDS`: Controls the core simulation speed. Lower values mean a faster simulation. For real-time speed, use `16000`.
* `SERVER_SLEEP_TIME_MICROSECONDS`: Controls how frequently the server sends data updates (e.g., to the frontend or AI). Lower values mean more frequent updates. For real-time updates, use `64000`.

**Build and run the server:**
*(Keep this terminal running)*

```bash
cargo build
cargo run
```

### 3. Set Up the Frontend (Web Interface)

This optional step sets up the web interface for visualizing the simulation and the AI's behavior.

**In a *new, separate terminal*, clone and prepare the frontend repository:**

```bash
# Make sure you are *outside* the rust-server directory first
# cd .. # If you are still inside rust-server

git clone https://github.com/outer-wilds-web/outer-wilds-front.git
cd outer-wilds-front
git checkout deep_learning # Switch to the required branch
```

**Configure frontend environment variables:**

Create a file named `.env` inside the `outer-wilds-front` directory:

```bash
echo "VITE_WEBSOCKET_URL=ws://localhost:3012" >> .env
```
* `VITE_WEBSOCKET_URL`: Specifies the address the frontend uses to connect to the Rust server's WebSocket.

**Install dependencies and run the frontend:**
*(Keep this terminal running)*

```bash
npm install
npm run dev
```

You should now be able to access the web interface, typically at `http://localhost:5173`.

### 4. Running HAL-9000

With the Rust server (and optionally the frontend) running, you can now run the HAL-9000 scripts from the initial Python environment terminal (where `.venv` is activated).

**Start Training:**

Replace `{name}` with the specific version/name of the model you want to train (e.g., `V0`).

```bash
python -m hal9000.model.Hal9000_2D_{name}
```

**Run Visualization/Inference:**

This script provides a lightweight visualization of the simulation and agent behavior. It's an alternative to using the full web frontend. Ensure the Rust server is running.

```Bash
python -m hal9000.display.display_ship2D
```

**Monitor Training Progress:**

Use TensorBoard to view logs and metrics generated during training. Run this command from the root directory of the HAL-9000 project.

```bash
tensorboard --logdir logs
```
Then navigate to the URL provided by TensorBoard (usually `http://localhost:6006`).

## Creating a New Training Agent

To experiment with different AI behaviors or reward structures:

1.  **Copy an existing model script:**
    Replace `{name}` with a unique identifier for your new agent (e.g., `V1`, `MyTest`).
    ```bash
    cp hal9000/model/Hal9000_2D_V0.py hal9000/model/Hal9000_2D_{name}.py
    ```
2.  **Modify the reward function:**
    Open the newly created file (`hal9000/model/Hal9000_2D_{name}.py`) and adjust the reward function logic according to your requirements.
3.  **Train your new agent:**
    Use the command from the "Start Training" section, replacing `{name}` with the identifier you chose.
