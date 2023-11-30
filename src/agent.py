import stable_baselines3
from stable_baselines3 import PPO
import os
import time
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import copy
from env import AeroEnv

steps = 10 #steps per episode used in model training before resetting

def make_env():
    def _init():
        env = AeroEnv(
            episodeSteps=steps, 
            numChords=3, 
            minChordLength=0.2,
            minChordSpacing=0.3,
            initialYSpacing=1, 
            initialChordLength=1, 
            initialUpperKulfan=1, 
            initialLowerKulfan=-0.3, 
            initialLEW=0.1, 
            initialN1=1, 
            initialN2=1, 
            dX_bounds = (-1, 1),
            dY_bounds = (-2, 2),
            dZ_bounds = (0, 0),
            dChord_bounds = (-0.5, 0.5),
            dTwist_bounds = (-5, 5),
            KT_bounds = (0, 1),
            KB_bounds = (-1, 0),
            N_bounds = (1, 3),
            LEW_bounds = (0, 0.6),
            kulfanWeightResolution = 5
        )
        return env
    return _init

models_dir = f"models/A2C-{int(time.time())}"
logdir = f"logs/A2C-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# envs = [make_env() for _ in range(3)]
# env = DummyVecEnv(envs)
# env = VecNormalize(env)
env = make_env()()
# evalEnv = copy.deepcopy(env)
# eval_callback = EvalCallback(
#     eval_env=evalEnv, 
#     best_model_save_path='./logs/best_model/',
#     log_path='./logs/eval_log/', 
#     eval_freq=1000)

# TODO: find a way to use eval callback that doesnt add 10 minutes per eval to the run

model = PPO(
	n_steps=steps,
    batch_size=steps,
	policy="MlpPolicy",
	env=env,
    n_epochs=10,
    learning_rate=.00025,
	verbose=0,
    tensorboard_log=logdir,
    stats_window_size=steps,
)

# if we want to avoid error message when running in DummyVecEnv, comment this line
env.render()

total_timesteps = 5000
model.learn(total_timesteps=total_timesteps)

print("We lerned")

num_episodes = 10
for _ in range(num_episodes):
    obs, info = env.reset()
    terminated = False
    for i in range(0,steps):
        print(i)
        action, nextHiddenState = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
    print("===\n\n\n\n")
    env.render()
    env.reset()
env.close()