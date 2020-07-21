from gridwrold_env import GridworldEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
import os

# Init Env
env = GridworldEnv(6, 4)
# Create log dir
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)
check_env(env, warn=True)
# Wrap it
env = make_vec_env(lambda: env, n_envs=1)

# Train the agent
model = ACKTR('MlpPolicy', env, verbose=1).learn(100)

# Test the trained agent
obs = env.reset()
n_steps = 40
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render()
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break


# Helper from the library
results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "DDPG LunarLander")