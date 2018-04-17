import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import re

agent = 'FrozenLake-v0'
env = gym.make(agent)
num_actions = list(map(int, re.findall('\d+', str(env.action_space))))[0]
num_states =  list(map(int, re.findall('\d+', str(env.observation_space))))[0]

print (num_states, num_actions)

gamma = 0.99
e = 0.1
episodes = 2000



tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, shape=(1,num_states))
targets = tf.placeholder(tf.float32, shape=(1, num_actions))
weights = tf.Variable(tf.random_uniform((num_states, num_actions), 0, 0.01))
outputs = tf.matmul(inputs, weights)
pred = tf.argmax(outputs, 1)

cost = tf.reduce_sum(tf.square(targets - outputs))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()

rewards_episode, timesteps = list(), list()

with sess:
	sess.run(init)
	for i in range(episodes):
		#env.render()
		s = env.reset()
		rewards = 0
		done = False
		timestep = 0
		while timestep < 99:
			timestep += 1
			action, Q = sess.run([pred, outputs], feed_dict={inputs:np.identity(num_states)[s:s+1]})
			if np.random.rand(1) < e:
				action[0] = env.action_space.sample()
			next_s, next_r, done, _ = env.step(action[0])
			next_Q = sess.run(outputs, feed_dict={inputs:np.identity(num_states)[next_s:next_s+1]})
			max_Q = np.max(next_Q)
			targetQ = Q
			targetQ[0, action[0]] = next_r + gamma * max_Q

			_, W = sess.run([optimizer, weights], feed_dict={inputs:np.identity(num_states)[s:s+1], targets:targetQ})
			rewards += next_r
			s = next_s	
		
			if done:
				print("Episode: {} finished after {} timesteps, Reward:{}".format(i + 1, timestep + 1, rewards))
				e = 1./ ((i/50) + 10)
				break

		rewards_episode.append(rewards)
		timesteps.append(timestep)

plt.plot(rewards_episode)
plt.show()
plt.plot(timesteps)
plt.show()			

print ('Successful episodes: {}'.format(str(np.sum(rewards)/ episodes)))

