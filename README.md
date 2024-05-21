
# Flappy Bird Deep Q-Network (DQN) Project

This project aims to implement Deep Q-Network (DQN) and its variants to solve the Flappy Bird game. It includes training scripts to train the models and a real-time testing script to observe the model's performance.

## Files

- **traindqn.py**: This script is used to train the DQN model.
- **trainddqn.py**: This script is used to train the Double DQN model.
- **traindueling.py**: This script is used to train the Dueling DQN model.
- **trainpri.py**: This script is used to train the PER DQN model.
- **test.py**: This script is used to test the trained models in real-time.

## Requirements

- Python 3.x
- Pygame (for running the Flappy Bird game)

## How to Run

1. Install the required dependencies:

```bash
pip install pygame
```

2. Train the models using the respective training scripts. For example:

```bash
python train_dqn.py
```

3. Once the models are trained, run the test script to observe the model's performance in real-time:

```bash
python test.py
```

## Note

- You may need to adjust the hyperparameters and training configurations in the training scripts according to your preferences and system specifications.
- Ensure that the Flappy Bird game environment is properly set up and accessible to the scripts.
