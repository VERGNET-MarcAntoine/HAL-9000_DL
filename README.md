# HAL-9000

## Overview
HAL-9000 is a collection of AI autopilots trained using reinforcement learning for integration with the Outer Wilds Web project. These AIs are designed to assist with autonomous navigation and decision-making within the game environment. The project is developed by Quentin Rollet, Marc-Antoine Vergnet, and Patrice Soulier, who also lead the development of Outer Wilds Web.


## Requirements
Ensure the following dependencies are installed:  
*(may work with other versions but not tested)*
- **Python 3.13.2** 
- **npm 11.1.0**
- **cargo 1.85.0**



## Quick Start
### Set Up the Rust Server
HAL-9000 interacts with the Outer Wilds Web Rust-based server. To set it up:
```bash
git clone https://github.com/outer-wilds-web/rust-server.git
cd rust-server && git checkout deep_learning
cargo build
cargo run
```


### Set Up the Frontend
In a separate terminal, configure the web interface:
```bash
git clone https://github.com/outer-wilds-web/outer-wilds-front.git
cd outer-wilds-front && git checkout deep_learning
echo "VITE_WEBSOCKET_URL=ws://localhost:3012" >> .env
npm install
npm run dev
```


### Setup Python environment
Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```


<!-- ## Version history

### v1.0.0 - First release

### v0.1.0 - First beta
Goals:
- Basic AI autopilot capable of simple pathfinding.
- First functional version of HAL-9000. -->
