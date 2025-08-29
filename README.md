# 🦖 Dino-RL

Train a reinforcement learning agent (PPO) to play a Chrome Dino–style game built with **PyGame** and **Stable-Baselines3**.  
The agent learns to **jump** and **duck** to avoid obstacles and survive as long as possible.

---

## 📦 Installation

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/masterA88/dino-rl.git
cd dino-rl

# create venv
python -m venv .venv

# activate (Windows)
.\.venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### ▶️ Train a new agent
```bash
python dino_rl.py --train --steps 500000
```
- Trains PPO on the PyGame environment for 500k timesteps (adjust as you like).
- A model file (e.g. `ppo_dino.zip`) is saved when training finishes.

### 🎮 Play with a trained agent
```bash
python dino_rl.py --play --model ppo_dino.zip
```
- Opens a PyGame window and the trained agent plays automatically.
- The terminal prints episode rewards (higher = better survival).

### 📊 Live metrics with TensorBoard
```bash
pip install tensorboard
tensorboard --logdir tb_dino
```
Open http://localhost:6006 and watch curves like:
- `rollout/ep_len_mean` (avg survival steps)
- `rollout/ep_rew_mean` (avg episode reward)
- `train/explained_variance`, `train/entropy_loss`, etc.

---

## 🧠 What’s inside

- **`DinoEnv`**: lightweight PyGame clone of the Dino game, exported as a **Gymnasium** env.  
  Observations are **84×84 grayscale images**; we **frame-stack(4)** for temporal context.  
- **PPO + CNN**: uses Stable-Baselines3 `CnnPolicy` with classic NatureCNN features.
- **Rewards**: +1 per step alive, −100 on crash, tiny penalty for action spam.

---

## 📊 Improvements & Tips

Make the agent stronger and training more stable:

1. **Do NOT use reward clipping**  
   Avoid wrapping with `ClipRewardEnv`. Keep full crash penalty (−100).

2. **Train longer with parallel envs**
   ```python
   from stable_baselines3.common.vec_env import SubprocVecEnv
   N = 8  # try 4 on smaller CPUs
   env = SubprocVecEnv([make_env() for _ in range(N)])
   ```
   Combine with frame-stack just like before.

3. **Better PPO hyperparameters**
   ```python
   model = PPO(
       "CnnPolicy", env,
       learning_rate=2.5e-4,
       n_steps=2048,          # larger rollout
       batch_size=512,        # multiple of n_envs
       n_epochs=4,
       gamma=0.997,
       gae_lambda=0.95,
       clip_range=0.2,
       ent_coef=0.005,
       vf_coef=0.4,
       tensorboard_log="./tb_dino/",
       seed=42,
       verbose=1,
   )
   ```

4. **Gentler difficulty ramp** (edit env)
   - Lower `base_speed` (e.g., 6.0)  
   - Slow the speed increase and widen spawn gaps (`spawn_min/max`) a bit.  
   This reduces early unwinnable patterns and stabilizes learning.

5. **Checkpoint while training**
   ```python
   from stable_baselines3.common.callbacks import CheckpointCallback
   ckpt = CheckpointCallback(save_freq=50_000, save_path="./checkpoints", name_prefix="ppo_dino")
   model.learn(total_timesteps=1_500_000, callback=ckpt, progress_bar=True)
   ```

6. **Evaluate across many episodes** (reduces variance)
   ```python
   from stable_baselines3.common.evaluation import evaluate_policy
   mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
   print(f"Mean reward over 20 eps: {mean_r:.1f} ± {std_r:.1f}")
   ```

---

## 📹 Record gameplay videos

Add a video recorder to your eval env:

```python
from stable_baselines3.common.vec_env import VecVideoRecorder

env = VecVideoRecorder(
    env,
    video_folder="videos",
    record_video_trigger=lambda step: step == 0,  # record first episode
    video_length=2000,
    name_prefix="dino_eval"
)
```

MP4 files will appear in the `videos/` folder (play with VLC or your default player).

---

## 🛠 Troubleshooting

- **Large files in git / push blocked**  
  Don’t commit your `.venv/`, logs, or binaries. Use a `.gitignore` like:
  ```
  .venv/
  __pycache__/
  tb_dino/
  videos/
  checkpoints/
  *.dll
  *.lib
  *.pyd
  ```
- **PyGame window doesn’t appear**  
  Update PyGame: `pip install --upgrade pygame`. Make sure you run `--play` from the repo root.
- **OpenCV DLL error on Windows**  
  Reinstall: `pip install --force-reinstall opencv-python==4.9.0.80`
- **Slow training**  
  Use `SubprocVecEnv` with 4–8 envs and consider a CUDA build of PyTorch if you have an NVIDIA GPU.

---

## 🙌 Contributing

Contributions are welcome!

1. Fork this repo  
2. Create a feature branch: `git checkout -b feature/my-improvement`  
3. Commit your changes: `git commit -m "feat: add X"`  
4. Push the branch: `git push origin feature/my-improvement`  
5. Open a Pull Request

---
