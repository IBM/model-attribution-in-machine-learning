# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
from tqdm import tqdm
import torch
import numpy as np
import os
#os.environ['TRANSFORMERS_CACHE'] = '/dccstor/secfl/mfoley/mlmat/.cache'
import inspect
import time
import random
from ppo_agent import PPOAgent
from envs.GymPayloadGenerationEnv import MLMATPayloadGeneration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, print_interval=10, save_interval=100):
    training_data = {'reward': [], 'loss': [], 'episode': []}
    training_reward = []
    agent = PPOAgent(input_dims, action_space, lr, betas, gamma, K_epochs, eps_clip)

    running_reward, time_step = 0, 0

    for i_episode in tqdm(range(1, max_episodes + 1)):
        state = env.reset()
        ep_reward = 0
        for t in range(max_timesteps):
            time_step += 1
            st = time.time()
            action = agent.get_action(state)
            ft = time.time() - st
            print(f'agent inference: {ft}')
            state, reward, done, _ = env.step(action)
            agent.store(reward, done)

            if time_step % update_timestep == 0:
                agent.train()
                agent.clear_memory()
                time_step = 0
            ep_reward += reward
            running_reward += reward
        # training_data['reward'].append(rep_reward)
        # training_data['episode'].append(i_episode)
        training_reward.append(ep_reward)

        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i_episode))
            torch.save(agent.policy.state_dict(), ckpt)
            print('Checkpoint saved')
            np.save(f'{ckpt_folder}/rewards.npy', np.array(training_reward))

        if i_episode % print_interval == 0:
            running_reward = int((running_reward / print_interval))
            print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
            running_reward = 0


if __name__ == '__main__':

    # set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # change checkpoint directory
    folder = '256-width-generation'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    base_model = 'opt-350m'
    ft_model = [str(i) for i in range(10)]

    config = {
        "base_model": base_model,
        'ft_models': ft_model,
        'max_steps': 20,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'vocab_dir': 'vocab/starting_vocab.txt'
    }
    env = MLMATPayloadGeneration(config)
    input_dims = env.observation_space.shape[1]

    action_space = [i for i in range(len(env.vocabulary))]
    print(f'action space: {len(action_space)}')
    print_interval = 50
    save_interval = 200
    max_episodes = 100000
    max_timesteps = 20
    # 200 episodes for buffer
    update_timesteps = 20000
    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002

    train(env, input_dims, action_space,
          max_episodes=max_episodes, max_timesteps=max_timesteps,
          update_timestep=update_timesteps, K_epochs=K_epochs,
          eps_clip=eps_clip, gamma=gamma, lr=lr,
          betas=[0.9, 0.990], ckpt_folder=ckpt_folder,
          print_interval=print_interval, save_interval=save_interval, )
