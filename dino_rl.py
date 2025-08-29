import os
import math
import random
import time
from typing import Tuple, Dict, Any

import cv2
import gymnasium as gym
import numpy as np
import pygame

from gymnasium.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor


# ----------------------------
# PyGame-based Dino clone ENV
# ----------------------------
class DinoEnv(gym.Env):
    """
    Minimal Chrome-Dino-like environment.
    - Observation: 84x84 grayscale image (later stacked via VecFrameStack).
    - Actions: 0 = do nothing, 1 = jump, 2 = duck.
    - Reward: +1 per alive step, -100 on collision, small penalties for useless jumping/ducking.
    Aim: survive as long as possible.
    """
    metadata = {"render_modes": ["human"]}
    def __init__(self, render_mode=None, fps=60, width=600, height=200, seed=None):
        super().__init__()
        self.width, self.height = width, height
        self.fps = fps
        self.clock = None
        self.render_mode = render_mode
        self.seed_val = seed

        # Dino physics
        self.ground_y = int(self.height * 0.8)
        self.dino_x = int(self.width * 0.1)
        self.dino_y = self.ground_y
        self.dino_w, self.dino_h = 34, 38

        # Jumping
        self.vel_y = 0.0
        self.jump_impulse = -10.5
        self.gravity = 0.7
        self.ducking = False
        self.duck_h = 24

        # Obstacles
        self.obstacles = []
        self.base_speed = 7.0
        self.speed = self.base_speed
        self.spawn_cooldown = 0
        self.spawn_min, self.spawn_max = 30, 70  # frames between spawns

        # Survival counter
        self.t = 0
        self.score = 0

        # Pygame surface
        self.surface = None

        # Gym API
        # 84x84 grayscale
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = Discrete(3)

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def _reset_pygame(self):
        if self.render_mode == "human":
            if not pygame.get_init():
                pygame.init()
            pygame.display.set_caption("DinoEnv")
            self.surface = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        else:
            if not pygame.get_init():
                pygame.init()
            self.surface = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()

    def seed(self, seed=None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def _spawn_obstacle(self):
        # Simple cactus: width ~ 20-30, height ~ 30-45; sometimes taller
        w = int(self.np_random.integers(18, 28))
        h = int(self.np_random.integers(28, 46))
        x = self.width + int(self.np_random.integers(0, 30))
        y = self.ground_y - h
        self.obstacles.append(pygame.Rect(x, y, w, h))

    def _step_physics(self, action):
        # Action:
        # 0 = do nothing, 1 = jump, 2 = duck
        reward = 1.0  # survival reward

        # Jump
        if action == 1 and self.dino_y >= self.ground_y:
            self.vel_y = self.jump_impulse
        # Duck
        self.ducking = (action == 2 and self.dino_y >= self.ground_y)

        # Simple action shaping: tiny penalty for spamming actions
        if action in (1, 2):
            reward -= 0.01

        # Apply gravity
        self.vel_y += self.gravity
        self.dino_y += self.vel_y

        # Ground collision
        if self.dino_y > self.ground_y:
            self.dino_y = self.ground_y
            self.vel_y = 0.0

        # Move obstacles
        for obs in self.obstacles:
            obs.x -= int(self.speed)

        # Remove off-screen
        self.obstacles = [o for o in self.obstacles if o.right > 0]

        # Spawn
        if self.spawn_cooldown <= 0:
            self._spawn_obstacle()
            self.spawn_cooldown = int(self.np_random.integers(self.spawn_min, self.spawn_max))
        else:
            self.spawn_cooldown -= 1

        # Speed ramps up slowly over time
        self.speed = self.base_speed + min(8.0, self.t / (60 * 12))  # up to +8 after ~12s

        # Collision check
        dino_h = self.duck_h if self.ducking else self.dino_h
        dino_rect = pygame.Rect(self.dino_x, int(self.dino_y - dino_h), self.dino_w, dino_h)
        done = any(dino_rect.colliderect(o) for o in self.obstacles)
        if done:
            reward -= 100.0

        return reward, done

    def _draw(self):
        self.surface.fill((255, 255, 255))  # white background

        # Ground line
        pygame.draw.line(self.surface, (200, 200, 200), (0, self.ground_y+1), (self.width, self.ground_y+1), 2)

        # Dino
        dino_h = self.duck_h if self.ducking else self.dino_h
        pygame.draw.rect(self.surface, (0, 0, 0),
                         (self.dino_x, int(self.dino_y - dino_h), self.dino_w, dino_h), 0)

        # Obstacles
        for o in self.obstacles:
            pygame.draw.rect(self.surface, (0, 0, 0), o, 0)

        # Score (optional)
        font = pygame.font.SysFont(None, 20)
        txt = font.render(f"Score: {int(self.score)}  Speed: {self.speed:.1f}", True, (50, 50, 50))
        self.surface.blit(txt, (8, 8))

        if self.render_mode == "human":
            pygame.display.flip()

    def _get_obs(self):
        # Convert the pygame surface to 84x84 grayscale
        raw = pygame.surfarray.array3d(self.surface)  # (W,H,3)
        frame = np.transpose(raw, (1, 0, 2))  # (H,W,3)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, None].astype(np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self._reset_pygame()

        # Reset state
        self.dino_y = self.ground_y
        self.vel_y = 0.0
        self.ducking = False
        self.obstacles = []
        self.speed = self.base_speed
        self.spawn_cooldown = 30
        self.t = 0
        self.score = 0

        # First draw
        self._draw()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        reward, done = self._step_physics(action)
        self.t += 1
        self.score += max(0, reward)

        self._draw()
        obs = self._get_obs()

        if self.render_mode == "human":
            self.clock.tick(self.fps)

        truncated = False  # no time limit here by default
        info: Dict[str, Any] = {"score": float(self.score)}
        return obs, reward, done, truncated, info

    def render(self):
        self.render_mode = "human"
        if self.surface is None:
            self._reset_pygame()
        self._draw()

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass


# ----------------------------
# Wrappers: grayscale/resize handled in-env; we still FrameStack
# ----------------------------
def make_env(render_mode=None, seed=42):
    def _f():
        env = DinoEnv(render_mode=render_mode, seed=seed)
        env = Monitor(env)
        # Optional: clip huge negative crash reward to stabilize learning a bit
        env = ClipRewardEnv(env)
        return env
    return _f


# ----------------------------
# Train + Evaluate
# ----------------------------
def train_and_eval(total_timesteps=500_000, model_path="ppo_dino.zip", render=False):

    # Single-process vec env + frame stack of 4 frames (common for vision)
    env = DummyVecEnv([make_env(render_mode="human" if render else None)])
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1,
        tensorboard_log="./tb_dino/",
        seed=42,
    )

    print("Training...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(model_path)
    env.close()
    print(f"Saved model to {model_path}")


# Evaluation (rendered)
# eval_env = DummyVecEnv([make_env(render_mode="human")])
# eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="last")

# obs = eval_env.reset()
# episode_reward = 0.0

# print("Evaluating (close the game window to stop)...")
# while True:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, rewards, dones, infos = eval_env.step(action)  # <-- 4 returns
#     episode_reward += float(rewards[0])
#     if dones[0]:
#         print(f"Episode reward: {episode_reward:.1f}")
#         episode_reward = 0.0
#         obs = eval_env.reset()


    # eval_env.close()  # (Unreachable due to simple loop; close window to quit)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train PPO on DinoEnv")
    parser.add_argument("--play", action="store_true", help="Run trained agent in the game")
    parser.add_argument("--model", type=str, default="ppo_dino.zip", help="Path to saved model")
    parser.add_argument("--steps", type=int, default=300_000, help="Train steps")
    args = parser.parse_args()

    if args.train:
        train_and_eval(total_timesteps=args.steps, model_path=args.model, render=False)

    if args.play:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

        env = DummyVecEnv([make_env(render_mode="human")])
        env = VecFrameStack(env, n_stack=4, channels_order="last")

        model = PPO.load(args.model, env=env, print_system_info=True)

        obs = env.reset()
        ep_rew = 0.0
        print("Playing (close the game window to stop)...")
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_rew += float(rewards[0])
            if dones[0]:
                print(f"Episode reward: {ep_rew:.1f}")
                ep_rew = 0.0
                obs = env.reset()
