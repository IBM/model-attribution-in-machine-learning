import matplotlib.pyplot as plt
import numpy as np

rewards = np.load('selection_rewards.npy')
base_rewards = np.load('selection_rewards_base.npy')
generation_rewards = np.load('generation_rewards.npy')
#rand_rewards = np.load('./rewards_selction.npy')

def get_moving_average(numbers, window_size=500):
    i = 0
    moving_averages = []
    while i < len(numbers) - window_size + 1:
        this_window = numbers[i : i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages

#rand_rewards = get_moving_average(rand_rewards, window_size=50)
rewards = get_moving_average(rewards, window_size=50)
base_rewards = get_moving_average(base_rewards, window_size=50)
generation_rewards = get_moving_average(generation_rewards, window_size=50)
#plt.plot(range(len(rand_rewards)), rand_rewards)
plt.plot(range(len(generation_rewards)), generation_rewards)
#plt.plot(range(len(base_rewards)), base_rewards)
#plt.plot(range(len(rewards)), rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
#plt.legend(['Generation FT','Selection Base', 'Selection FT'])
plt.savefig('./training_prompt_generation.eps', format='eps')
#plt.show()