import argparse
import torch  # Additional

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder

import gymnasium as gym

from gymnasium.wrappers import TimeLimit
import wandb
from wandb.integration.sb3 import WandbCallback

import humanoid_bench
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from collections import deque

# [Additional] Setting up devices for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Setting up the argument parser to take command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str)
parser.add_argument("--seed", type=int)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--learning_rate", default=3e-5, type=float)
parser.add_argument("--max_steps", default=20000000, type=int)
parser.add_argument("--wandb_entity", default="robot-learning", type=str)
ARGS = parser.parse_args()


# Function to create a new environment with specified parameters
def make_env(
    rank,
    seed=0
):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    def _init():
        # Initialize environment
        env = gym.make(ARGS.env_name)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        # Set action space speed
        env.action_space.seed(ARGS.seed + rank)
        
        return env

    return _init

# Custom callback to log additional information in TensorBoard
class EvalCallback(BaseCallback):
    
    def __init__(self, eval_every: int = 100000, verbose: int = 0):
        super(EvalCallback, self).__init__(verbose=verbose)

        self.eval_every = eval_every
        self.eval_env = DummyVecEnv([make_env(ARGS.num_envs)])

    def _on_step(self) -> bool:
        # Check if it's time to record a video
        if self.num_timesteps % self.eval_every == 0:
            self.record_video()

        return True
    
    def record_video(self) -> None:
        # Record and log video
        print("recording video")
        video = []

        obs = self.eval_env.reset()
        for i in range(1000):
            action = self.model.predict(obs, deterministic=True)[0]
            obs, _, _, _ = self.eval_env.step(action)
            pixels = self.eval_env.render().transpose(2,0,1)
            video.append(pixels)

        video = np.stack(video)
        wandb.log({"results/video": wandb.Video(video, fps=100, format="gif")})


# Custom callback to log episode-level information
class LogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.aux_rewards = {}
        self.aux_returns = {}
        for key in info_keywords:
            self.aux_rewards[key] = np.zeros(ARGS.num_envs)
            self.aux_returns[key] = deque(maxlen=100)


    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for idx in range(len(infos)):
            # Accumulate rewards
            for key in self.aux_rewards.keys():
                self.aux_rewards[key][idx] += infos[idx][key]

            if self.locals['dones'][idx]:
                # Record returns and reset rewards
                for key in self.aux_rewards.keys():
                    self.aux_returns[key].append(self.aux_rewards[key][idx])
                    self.aux_rewards[key][idx] = 0
        return True

    def _on_rollout_end(self) -> None:
        # Log auxiliary returns
        for key in self.aux_returns.keys():
            self.logger.record("aux_returns_{}/mean".format(key), np.mean(self.aux_returns[key]))


# Episode logging callback class
class EpisodeLogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, info_keywords=()):
        super().__init__(verbose)
        self.returns_info = {
            "results/return": [],
            "results/episode_length": [],
            "results/success": [],
            "results/success_subtasks": [],
        }


    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        print(infos, "################")
        for idx in range(len(infos)):
            curr_info = infos[idx]
            if "episode" in curr_info:
                # Log episode information
                self.returns_info["results/return"].append(curr_info["episode"]["r"])
                self.returns_info["results/episode_length"].append(curr_info["episode"]["l"])
                
                cur_info_success = 0
                if "success" in curr_info:
                    cur_info_success = curr_info["success"]
                self.returns_info["results/success"].append(cur_info_success)
                
                cur_info_success_subtasks = 0
                if "success_subtasks" in curr_info:
                    cur_info_success_subtasks = curr_info["success_subtasks"]
                self.returns_info["results/success_subtasks"].append(cur_info_success_subtasks)
        return True

    def _on_rollout_end(self) -> None:
        # Log episode information
        for key in self.returns_info.keys():
            if self.returns_info[key]:
                self.logger.record(key, np.mean(self.returns_info[key]))
                self.returns_info[key] = []


def main(argv):
    env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])  # Create multiple environments for parallel training
    
    steps = 1000

    # Initialize Weights and Biases logging    
    run = wandb.init(
        entity=ARGS.wandb_entity,
        project="humanoid-bench",
        name=f"ppo_{ARGS.env_name}",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        config={"device": "cuda" if torch.cuda.is_available() else "cpu"}  # [additional] add device
    )
    
    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", learning_rate=float(ARGS.learning_rate), batch_size=512, device=device)  # Add "device=device"
    model.policy.to(device)  # Move the model parameters to the GPU
    model.learn(total_timesteps=ARGS.max_steps, log_interval=1, callback=[WandbCallback(model_save_path=f"models/{run.id}",verbose=2), EvalCallback(), LogCallback(info_keywords=[]), EpisodeLogCallback()])
    
    print(f"Model device: {next(model.policy.parameters()).device}")
    
    # Save the model
    model_save_path = f"models/{run.id}/ppo"
    model.save("ppo")
    print(f"Saving model to {model_save_path}")
    print("Training finished")

if __name__ == '__main__':
#   app.run(main)
    main(None)
