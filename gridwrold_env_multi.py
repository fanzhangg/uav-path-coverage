import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
from enum import IntEnum
from itertools import product
from tabulate import tabulate


class Board:
    """Represent a grid and operations on it"""
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.data = np.zeros((height, width), dtype=np.int)

        self.total_grids = width * height
        self.visited_grids = 0

    def set(self, i, j):
        """
        Increment the visited counts in the grid
        :param i: row
        :param j: column
        """
        assert i >= 0 and i < self.height
        assert j >= 0 and j < self.width

        if self.data[i, j] == 0:    # Not visited
            self.visited_grids += 1

        self.data[i, j] += 1

        return self.data[i, j]

    def get(self, i, j):
        """
        Increment the visited counts in the grid
        :param i: row
        :param j: column
        :return:
        """
        assert i >= 0 and i < self.height
        assert j >= 0 and j < self.width
        return self.data[i, j]

    def is_valid(self, i, j):
        """Check if a position is in the boundary"""
        return 0 <= i < self.height and 0 <= j < self.width

    def is_filled(self):
        """
        :return: True if the board is filled, otherwise false
        """
        for row in self.data:
            for i in row:
                if i == 0:
                    return False    # Not filled
        return True

    def __str__(self):
        return str(self.data)


class GridworldEnv(gym.Env):
    """
    Gridworld Environment that represents a rectangle world
    """

    metadata = {'render.modes': ['human']}

    class Actions(IntEnum):
        left = 0
        right = 1
        up = 2
        down = 3

    def __init__(self, width, height, seed=1337):
        super(GridworldEnv, self).__init__()
        # self.world = np.array((width, height), dtype=int)

        # Environment configuration
        self.width = width
        self.height = height
        self.size = width * height

        # Information for the agent
        self.agent_positions = None
        self.all_steps = None
        self.board = Board(width, height)

        # For gym
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(16)
        # Observations are number of cells
        self.observation_space = spaces.Box(low=0, high=max((width, height)),
                                            shape=(4, ), dtype=np.int)

        # Initialize the state
        self.reset()

        # Initialize the RNG
        self.seed(seed=seed)

        # Action enumeration for this environment
        actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.action_combs = [p for p in product(actions, repeat=2)]     # All possible actions

    def reset(self):
        # Current position and direction of the agent
        self.agent_positions = [(0, 0), (self.height - 1, self.width - 1)]

        self.board = Board(self.width, self.height)
        self.board.set(0, 0)
        self.board.set(self.width - 1, self.height - 1)

        # Step count since episode start
        self.all_steps = [[(0, 0)], [(self.width - 1, self.height - 1)]]

        # Return first observation
        return np.array([0, 0, self.height - 1, self.width - 1])

    def seed(self, seed=1337):
        """
        Seed the random number generator
        """
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def get_obs_space(self):
        """
        :return: The agent's positions as a 1D np array
        """
        obs = []
        for pos in self.agent_positions:
            obs.extend(pos)
        return np.array(obs)

    def step(self, action):
        action_comb = self.action_combs[action]
        agent_positions = []
        is_visited = False

        for index, a in enumerate(action_comb):
            prev_i, prev_j = self.agent_positions[index]
            i, j = prev_i + a[0], prev_j + a[1]
            if not self.board.is_valid(i, j):   # New position out of bound
                # Skip this step, has a -1 reward
                return self.get_obs_space(), -1, False, {}
            elif self.board.get(i, j) > 0:  # Revisit a grid
                # Terminate the episode, has no reward
                is_visited = True
            agent_positions.append((i, j))

        # Update positions
        self.agent_positions = agent_positions

        for index, pos in enumerate(self.agent_positions):
            i, j = pos
            self.board.set(i, j)    # Update pos
            self.all_steps[index].append(pos)   # Update steps

        if is_visited:  # Is visited
            return self.get_obs_space(), 0, True, {}
        elif self.board.is_filled():  # All grids has been visited once
            return self.get_obs_space(), 100000, True, {}
        else:   # The grid has not been visited
            return self.get_obs_space(), 1, False, {}

    def render(self, mode='human', close=False):
        print("board:")
        print(self.board.data)
        print("path:", self.get_path())
        print("pos:", self.agent_positions)
        print("")

    def get_path(self):
        """
        Get the path on the field
        :return:
        """
        tables = ""
        for steps in self.all_steps:
            board = np.zeros((self.height, self.width), dtype=np.int)
            for index, pos in enumerate(steps):
                i, j = pos
                board[i, j] = index
            table = tabulate(board)
            tables += "\n"
            tables += table
        return tables

