import numpy as np
import gym
from gym import spaces
from stable_baselines.common.env_checker import check_env


class Board:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = np.zeros((self.rows, self.cols), dtype=np.int)

    def get(self):
        self.data[]


class GoLeftEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}
    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1
    TOP = 2
    DOWN = 3

    def __init__(self, rows=4, cols=6):
        super(GoLeftEnv, self).__init__()

        # Size of the 1D-grid
        self.grid_size = rows * cols
        # Initialize the agent at the left top of the grid
        self.agent_pos = (0, 0)

        self.board = np.zeros((rows, cols), dtype=np.int)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=self.grid_size,
                                            shape=(1,), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the agent at the right of the grid
        self.agent_pos = (0, 0)
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return (self.agent_pos, self.board)

    def step(self, action):
        i, j = self.agent_pos
        if action == self.LEFT:
            self.agent_pos = i, j - 1
        elif action == self.RIGHT:
            self.agent_pos += i, j + 1
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        done = bool(self.agent_pos == 0)

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.agent_pos == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return np.array([self.agent_pos]).astype(np.float32), reward, done, info

    def render(self, mode='console'):
        # agent is represented as a cross, rest as a dot
        print("." * self.agent_pos, end="")
        print("x", end="")
        print("." * (self.grid_size - self.agent_pos))
        print("Observation: ", np.array([self.agent_pos]).astype(np.float32))

    def close(self):
        pass


from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env

# Instantiate the env
env = GoLeftEnv(grid_size=10)
# wrap it
env = make_vec_env(lambda: env, n_envs=1)

# Train the agent
model = ACKTR('MlpPolicy', env, verbose=1).learn(000)

# Test the trained agent
obs = env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render(mode='console')
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
