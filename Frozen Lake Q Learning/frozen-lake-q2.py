import gym
import numpy as np
import matplotlib.pyplot as plt
import re

np.random.seed(28)

agent = 'FrozenLake-v0'
env = gym.make(agent)
num_actions = list(map(int, re.findall('\d+', str(env.action_space))))[0]
num_states =  list(map(int, re.findall('\d+', str(env.observation_space))))[0]

print (num_states, num_actions)

Q = np.zeros((num_states, num_actions))

episodes = 1000
learning_rate = 0.8
max_timesteps = 99
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

timesteps, rewards = list(), list()

for episode in range(episodes):
	state = env.reset()
	done = False
	reward = 0
	for step in range(max_timesteps):
		e = np.random.uniform(0, 1)
		if e > epsilon:
			action = np.argmax(Q[state, :])
		else:
			action = env.action_space.sample()

		next_state, next_reward, done, _ = env.step(action)
		
		Q[state, action] += learning_rate * (next_reward + gamma * np.max(Q[next_state, :]) - Q[state, action])	
		reward += next_reward
		state = next_state

		if done:
			print("Episode: {} finished after {} timesteps, Reward:{}".format(episode + 1, step + 1, reward))
			break

	episode += 1		
	epsilon = min_epsilon + (max_epsilon - min_epsilon)* np.exp(-decay_rate*episode)		
	rewards.append(reward)
	timesteps.append(step)

print (Q)
plt.plot(rewards)
plt.show()
plt.plot(timesteps)
plt.show()			

print ('Successful episodes: {}'.format(str(np.sum(rewards)/ episodes)))


##### Playing using Q Table

env.reset()
for episode in range(1):
	state = env.reset()
	step = 0
	done = False
	print ("-----***-----")
	print ('Episode', episode)
	for step in range(max_timesteps):
		env.render()
		action = np.argmax(Q[state, :])
		next_state, next_reward, done, _ = env.step(action)
		if done:
			break
		state = next_state
		
env.close()							