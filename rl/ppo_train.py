# rl/ppo_train.py
import argparse, os
import gym                         # pip install "gym==0.25.2"
import crafter                     # pip install crafter
from stable_baselines3 import PPO  # pip install --no-deps "stable-baselines3==1.8.0"
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

def make_env(logdir, env_id="CrafterReward-v1", seed=0):
    env = gym.make(env_id)
    env = crafter.Recorder(env, logdir, save_stats=True, save_video=False, save_episode=False)
    env.seed(seed)
    env = Monitor(env)
    return env

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="results/ppo")
    ap.add_argument("--total-steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-id", default="CrafterReward-v1")
    ap.add_argument("--save-freq", type=int, default=100_000)
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    env = make_env(os.path.join(args.logdir, "train"), args.env_id, args.seed)
    eval_env = make_env(os.path.join(args.logdir, "eval"), args.env_id, args.seed + 1)

    ckpt = CheckpointCallback(save_freq=args.save_freq, save_path=args.logdir, name_prefix="ppo_ckpt")
    eval_cb = EvalCallback(eval_env, best_model_save_path=args.logdir, eval_freq=10_000,
                           n_eval_episodes=10, deterministic=True, render=False)

    model = PPO("CnnPolicy", env, verbose=1, seed=args.seed,
                tensorboard_log=os.path.join(args.logdir, "tb"))

    model.learn(total_timesteps=args.total_steps, callback=[ckpt, eval_cb])
    model.save(os.path.join(args.logdir, "final_model"))

if __name__ == "__main__":
    main()
