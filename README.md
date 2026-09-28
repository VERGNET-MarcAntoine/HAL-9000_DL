# HAL-9000

## Overview
HAL-9000 is a collection of AI autopilots trained using reinforcement learning for integration with the Outer Wilds Web project. These AIs are designed to assist with autonomous navigation and decision-making within the game environment. The project is developed by Quentin Rollet, Marc-Antoine Vergnet, and Patrice Soulier, who also lead the development of Outer Wilds Web.

## Requirements

Ensure you have the following dependencies installed. While other versions might work, they have not been tested:

* **[uv](https://docs.astral.sh/uv/):** manages Python and the dependencies (installs Python 3.13 automatically if needed)
* **npm:** 11.1.0
* **Cargo (Rust):** 1.85.0

## Quick Start Guide

Follow these steps to set up the environment and run the HAL-9000 agents.

### 1. Setup Python Environment (HAL-9000 Core)

This sets up the environment for the AI training and execution scripts.

**Create the virtual environment and install dependencies:**

```bash
uv sync
```

This creates a `.venv` in the project directory from the locked versions in `uv.lock` — nothing is installed in your global Python. Every command below is prefixed with `uv run`, so there is no need to activate the environment.

To update the dependencies later: `uv lock --upgrade && uv sync`.

**Configuration:**

All settings live in `config.toml` at the root of the project:

```toml
[simulation]
speedup = 1               # 1 = real time (to watch a model), e.g. 25 to train faster
decision_interval = 0.25  # simulated seconds between two decisions of the AI

[server]
websocket_url = "ws://127.0.0.1:3012"
path = "rust-server"

[training]
episode_time = 15         # maximum episode duration, in simulated minutes
number_episode = 15000    # total number of training episodes
save_number = 100         # save the model every N episodes
```

**Simulation speed and synchronization:** the Rust server advances the simulation by a fixed 1/60 s tick, so it can be accelerated without changing the physics. The Python environment must follow the same acceleration, otherwise each step covers more or less simulated time than during training and a model behaves differently when replayed. Both sides are therefore derived from `speedup`:

* Python waits `decision_interval / speedup` seconds of real time between two steps.
* The server is launched with `SIMULATION_SLEEP_TIME_MICROSECONDS = 16667 / speedup` and `SERVER_SLEEP_TIME_MICROSECONDS = 4 × SIMULATION_SLEEP_TIME_MICROSECONDS` (see below).

Keep the same `decision_interval` to train and to replay a model. Note: `MAGB_V0` was trained at about 0.13 s per decision.

### 2. Set Up the Rust Server (Outer Wilds Web Simulation)

HAL-9000 interacts with the Rust-based server that runs the Outer Wilds Web simulation.

**Clone the server repository into the HAL-9000 directory** (it is ignored by git):

```bash
git clone -b deep_learning https://github.com/outer-wilds-web/rust-server.git
```

Then remove the line `rdkafka = ...` from `rust-server/Cargo.toml`: this dependency is not used by the code and requires `cmake` to build.

**Run the server:**
*(Keep this terminal running)*

```bash
uv run python -m hal9000.server
```

This builds the server in release mode and launches it with the speed and address set in `config.toml`. The server only listens on the host of `websocket_url` (`127.0.0.1` by default).

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

With the Rust server (and optionally the frontend) running, you can now run the HAL-9000 scripts from the root directory of the HAL-9000 project.

**Start Training:**

Training runs on an accelerated simulation, with several ships trained in parallel on the same server (`[training]` section of `config.toml`: `speedup = 25` and `n_envs = 6` by default, about 500 steps per second). Start the server at the training speed:

```bash
uv run python -m hal9000.server --train
```

Then, in another terminal, replace `{name}` with the specific version/name of the model you want to train (e.g., `V0`):

```bash
uv run python -m hal9000.model.Hal9000_2D_{name}
```

Training resumes from the latest saved model of the same name in `models/` if there is one. Before starting, the script measures the actual speed of the server and stops if it does not match the training configuration. During training, a warning is printed if Python cannot keep up with the simulation: lower `speedup` or `n_envs` in that case. The policy runs on CPU, which is faster than GPU for this small network.

**Evaluate a Trained Model:**

With the server running at real-time speed (`uv run python -m hal9000.server`), runs 10 episodes with one of the pre-trained models from `models/` and prints the total reward of each episode.

```bash
uv run python -m hal9000.test.test_MAGB_V1   # or test_MAGB_VO, test_PAT_V1
```

**Run Visualization/Inference:**

This script provides a lightweight visualization of the simulation and agent behavior. It's an alternative to using the full web frontend. Ensure the Rust server is running. It shows every ship connected to the server, so it can also be used to watch the ships during training.

```bash
uv run python -m hal9000.display.display_ship2D
```

**Monitor Training Progress:**

Use TensorBoard to view logs and metrics generated during training (`rollout/ep_rew_mean` is the mean reward per episode). Run this command from the root directory of the HAL-9000 project.

```bash
uv run tensorboard --logdir logs
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

## Security Notes

* **Only load models you trust.** Stable-Baselines3 `.zip` files contain objects serialized with pickle, and loading a model can execute arbitrary code. The scripts load models through `hal9000.model.core.loading.load_ppo`, which avoids unpickling the saved learning-rate and clip-range functions (this also fixes crashes when loading the models with a different Python version than the one they were trained with), but other pickled objects remain.
* The WebSocket connection to the Rust server is unencrypted and unauthenticated (`ws://`). It is meant for local use only: do not expose the server port to an untrusted network.
