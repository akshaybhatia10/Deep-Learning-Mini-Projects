import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
print('Actions: ' + str(env.action_space), env.action_space.n)
print ('Observations: ' + str(env.observation_space), env.observation_space.n)

# Initialize table with zeros
Q = np.zeros((env.observation_space.n, env.action_space.n))

learning_rate = 0.8
gamma = 0.95
episodes = 1000

rewards = []

for i in range(episodes):
	s = env.reset()
	env.render()
	rAll = 0
	timestep = 0
	while timestep < 99:
		timestep += 1
		a  = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
		s1, reward, done, info = env.step(a)
		Q[s, a] += learning_rate * (reward + gamma*np.max(Q[s1, :]) - Q[s,a])
		rAll += reward
		s = s1
		print (s)
		if done:
			#print("Episode finished after {} timesteps, Reward:{}".format(i+1, reward))
			break
	rewards.append(rAll)

print ('Score : {}'.format(str(sum(rewards)/ episodes)))
print ('Q Table: {}'.format(Q))

plt.plot(rewards)
plt.show()			 
