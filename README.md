# Project
Project for the course "Reinforcement Learning" - UniTS.

## Description
This project consists in implementing the board game [Risk](https://en.wikipedia.org/wiki/Risk_(game)) as a fully interactive environment through the use of PettingZoo, then trains a collection of agents through the use of **Multi-Agent Reinforcement Learning** techniques.<br>
The goal is to explore how different learning algorithms behave when multiple autonomous agents compete on the same game board, to construct a potential solution to Risk games and to understand the different approaches to the problem.

## Map Configuration
The Risk environment is *data-driven*, it does not hard-code any specific world graph.<br>
Instead, the game world is described by a **JSON map file** placed in the `environment/maps/` directory. This is on purpose, since it allows for easier testing on differently-sized environments. The standard default Risk environment is already implemented in the `environment/classic.json`, and will be the default one if no other map is provided.

## Running the Project

1. **Set up a virtual environment** (or conda) and activate it:  
   ```bash
   source .venv/bin/activate          # for venv  
   # or  
   conda activate <env_name>          # for conda
   ```

2. **Install dependencies**  
   - Using *pip*: `pip install -r requirements.txt`  
   - Or with *uv*: `uv sync && uv lock` (choose the method you prefer).  
   - If you have a GPU, make sure the appropriate `torch` build is installed before running the above commands.

3. **Run the training / evaluation script**  
   ```bash
   python train_watch_PPO.py
   ```
   The script defines, among many others, two Boolean arguments with the following default values:

   - `train=False` – training is disabled by default. Set to `True` to train the agents from scratch (this can be time‑consuming; you may speed it up by using a smaller map such as `environment/simplified.json` or `environment/africa.json`, or by reducing the network size).  
   - `watch=True` – visualisation is enabled by default. When `True`, the script loads the latest trained models and displays five independent matches (a trained model must exist for this mode to work).

   Adjust these arguments in the script to enable the desired behavior.<br>
   NOTE: Changing the model's hyperparameters will unavoidably require re-training!