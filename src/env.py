import numpy as np
import copy

class Env2048():
    def __init__(self):
        pass

    def num_free_pos(self, s):
        '''Gives the number of "free" positions (0-valued environment positions) on the game board'''
        return sum([1 for row in range(s.shape[0]) for col in range(s.shape[1]) if s[row, col] == 0])

    def add_new_num(self, s):
        '''Adds a new number at an open game board position''' 
        # Find all open positions and choose one
        valid_states = np.argwhere(s == 0)
        idx = valid_states[np.random.randint(len(valid_states))].tolist()

        # Insert new 2 or 4 randomly at chosen location
        new_val = 2 if np.random.rand() < 0.5 else 4
        s[idx[0], idx[1]] = new_val

        return s


    def move(self, s, a):
        '''Moves state based on action'''

        def move_left(s):
            def shift_zeros(s):
                # Move all zeros before combining
                for row in range(rows):
                    for col in range(cols - 1):
                        # If empty, move left. If same, combine
                        if s[row, col] == 0:
                            s[row, col], s[row, col + 1] = s[row, col + 1], s[row, col]
                return s
            r = 0
            rows, cols = s.shape
            for _ in range(4):
                s = shift_zeros(s)

            for row in range(rows):
                for col in range(1, cols):
                    if s[row, col - 1] == s[row, col]:
                        s[row, col - 1] *= 2
                        s[row, col] = 0
                        r += s[row, col - 1]
                        s = shift_zeros(s)

            return s, r

        def move_right(s):
            def shift_zeros(s):
                # Move all zeros before combining
                for row in range(rows):
                    for col in range(cols-1,0,-1):
                        # If empty, move right. If same, combine
                        if s[row, col] == 0:
                            s[row, col], s[row, col - 1] = s[row, col - 1], s[row, col]
                return s
            r = 0
            rows, cols = s.shape
            for _ in range(4):
                s = shift_zeros(s)

            for row in range(rows):
                for col in range(cols - 2, -1, -1):
                    if s[row, col + 1] == s[row, col]:
                        s[row, col + 1] *= 2
                        s[row, col] = 0
                        r += s[row, col + 1]
                        s = shift_zeros(s)

            return s, r

        def move_up(s):
            def shift_zeros(s):
                # Move all zeros before combining
                for row in range(rows - 1):
                    for col in range(cols):
                        # If empty, move up. If same, combine
                        if s[row, col] == 0:
                            s[row, col], s[row + 1, col] = s[row + 1, col], s[row, col]
                return s
            r = 0
            rows, cols = s.shape

            for _ in range(4):
                s = shift_zeros(s)

            for row in range(1, rows):
                for col in range(cols):
                    if s[row - 1, col] == s[row, col]:
                        s[row - 1, col] *= 2
                        s[row, col] = 0
                        r += s[row - 1, col]
                    s = shift_zeros(s)

            return s, r

        def move_down(s):
            def shift_zeros(s):
                # Move all zeros before combining
                for row in range(rows-1,0,-1):
                    for col in range(cols):
                        # If empty, move right. If same, combine
                        if s[row, col] == 0:
                            s[row, col], s[row - 1, col] = s[row - 1, col], s[row, col]
                return s
            r = 0
            rows, cols = s.shape
            for _ in range(4):
                s = shift_zeros(s)

            for row in range(rows - 2, -1, -1):
                for col in range(cols):
                    if s[row + 1, col] == s[row, col]:
                        s[row + 1, col] *= 2
                        s[row, col] = 0
                        r += s[row + 1, col]
                        s = shift_zeros(s)

            return s, r

        if a == 0:
            return move_left(s)
        elif a == 1:
            return move_right(s)
        elif a == 2:
            return move_up(s)
        elif a == 3:
            return move_down(s)
        else:
            raise ValueError('Incorrect action passed to env.move()')

    def step(self, s, a):
        '''     
        Given a state and action, returns the reward for that action along with a sample of the next state

        Args:
        - s (list): state in the form [dealer card, player sum]
        - a (int): action to take (0,1,2,3 for left, right, up, down)
        Returns:
        - s_prime: sample of the next state of the environment after the current action is taken
        - reward: value gained from performing action in the environment
        '''
        # Take action and observe next state
        s_prime, reward = self.move(s.copy(), a)

        if np.not_equal(s, s_prime).any(): # action made change
            s_prime = self.add_new_num(s_prime)

        return s_prime, reward