from env import Env2048
import numpy as np

if __name__ == '__main__':
    env = Env2048()
    action_desc_dict = {0: 'Left', 1: 'Right', 2: 'Up', 3: 'Down'}
    states_actions = [(np.array([[2, 4, 16, 0], 
                                 [2, 0, 0, 0], 
                                 [2, 0, 0, 0], 
                                 [2, 4, 8, 0]]), 3),
                        (np.array([[2, 0, 0, 0], 
                                    [0, 2, 0, 0], 
                                    [0, 0, 2, 0], 
                                    [2, 0, 0, 2]]), 0),
                        (np.array([[2, 0, 0, 4], 
                                    [2, 0, 0, 8], 
                                    [2, 0, 0, 16], 
                                    [2, 0, 0, 32]]), 1),
                        (np.array([[0, 0, 0, 0], 
                                    [0, 0, 0, 0], 
                                    [2, 2, 2, 2], 
                                    [2, 2, 2, 2]]), 2)]

    for idx, (state, action) in enumerate(states_actions):
        action_desc = action_desc_dict[action]
        state_prime, reward = env.step(state, action)

        # Output state stuff
        print('State ', idx + 1)
        print(state)
        print('Moving ', action_desc)
        print(state_prime)
        print('Reward: ', reward)
        print()