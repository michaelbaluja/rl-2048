# Reinforcement Learning for 2048
This project utilizes reinforcement learning techniques with neural network-based non-linear function approximation to play the game [2048](https://2048game.com/). 

## Installing
To install, clone git repository or download compressed file & unpack in your desired development environment.

```bash
git clone https://github.com/michaelbaluja/rl-2048.git
```

## Running
To run the code for this repository, the `main.ipynb` notebook has been added with all necessary imports to run. 
1. Open the `main.ipynb` notebook 
2. Run the import and hyperparameter cell
3. (Optionally) Run the warning ignore cell
4. Run the relevant algorithm cells (Monte Carlo/SARSA) 

Each cell fully contains the required calls to create and run the agent (except for the previously noted hyperparameters), and plot output. Note that the default calls to the output plotting function saves each plot. This can be turned off by removing the `save_file='...'` parameter to the plotter function in each cell.
