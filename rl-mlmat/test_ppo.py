# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py
from tqdm import tqdm
import torch
import numpy as np
import os
import inspect
import random
from ppo_agent import PPOAgent
from envs.GymPayloadSelectionEnv import MLMATPayloadSelection
#from envs.GymPayloadSelectionEnvBase import MLMATPayloadSelection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, print_interval=10, save_interval=100):


    training_data = {'reward': [], 'loss':[], 'episode': []}
    training_reward = []
    predictions = []
    agent = PPOAgent(input_dims, action_space, lr, betas, gamma, K_epochs, eps_clip)
    rewards_step = [] 
    running_reward, time_step = 0, 0
    
    for i_episode in tqdm(range(1, max_episodes + 1)):
        state = env.reset()
        predictions = []
        ep_reward = 0
        previous_reward = -1
        env.ft = i_episode  - 1
        for t in range(max_timesteps):
            time_step += 1
            #action = random.choice(action_space)#agent.get_action(state)
            #print('using random agent')
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
            agent.store(reward, done)
            rewards_step.append(reward)
            predictions.append(1 if reward > previous_reward else 0)
            if time_step % update_timestep == 0:
                agent.clear_memory()
                time_step = 0
            print(env.prompt)
            print(reward)
            print(info)
            ep_reward += reward
            previous_reward = reward
            running_reward += reward
        #training_data['reward'].append(rep_reward)
        #training_data['episode'].append(i_episode)
        training_reward.append(ep_reward)
        if i_episode % print_interval == 0:
            print(predictions)
            running_reward = int((running_reward / print_interval))
            print('Episode {} \t Avg reward: {}'.format(i_episode, running_reward))
            print('Episode {} \t Correct Predictions: {}/{}'.format(i_episode, sum(predictions), max_timesteps*print_interval))
            running_reward = 0
        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i_episode))
            #torch.save(agent.policy.state_dict(), ckpt)
            #np.save(f'{ckpt_folder}/random_rewards.npy', np.array(training_reward))


if __name__ == '__main__':

    # set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # change checkpoint directory
    folder = '256-width-Selection'
    ckpt_folder = os.path.join(os.getcwd(), "Models", folder)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)

    base_model = 'opt-350m'
    ft_model = [str(i) for i in range(10)]
    #ft_model = [str(i) for i in range(10,20)]


    config = {
        "base_model": base_model,
        'ft_models': ft_model,
        'max_steps': 20,
        'device': 'cuda',
        '0-9': True,
        }
    env = MLMATPayloadSelection(config)
    input_dims = env.observation_space.shape[1]
    print(env.ft_models)
    action_space = [i for i in range(len(env.prompts))]
    print_interval = 1
    save_interval = 200
    max_episodes = 10
    max_timesteps = 20
    # 200 episodes for buffer
    update_timesteps = 20000
    K_epochs = 6
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.002


    test(env, input_dims, action_space,
              max_episodes=max_episodes, max_timesteps=max_timesteps,
              update_timestep=update_timesteps, K_epochs=K_epochs,
              eps_clip=eps_clip, gamma=gamma, lr=lr,
              betas=[0.9, 0.990], ckpt_folder=ckpt_folder,
              print_interval=print_interval, save_interval=save_interval, )
