from Architechture import Node, Edge, Graph
from random import sample, seed, randint
import numpy as np
import tensorflow as tf
from time import sleep

seed(69)
np.random.seed(69)

def createGraph():
    graph = Graph()
    graph.addNodes([Node(index) for index in range(10)])
    nodes = graph._nodes
    graph.addEdges([Edge(nodes[source], nodes[destination]) for [source, destination] in [sample(range(10), 2) for _ in range(40)]]) 
    return graph

def startEpisode(graph): return tuple(sample(graph._nodes, 2))
def convertState(state): return [int(state[0].__name__), int(state[1].__name__)]

def processRewards(rewards, gamma = 0.95):
	for iteration in range(rewards.__len__()):
		rewards[iteration] = sum(list(map(lambda x: x[1]*(gamma**x[0]), enumerate(rewards[iteration:]))))
	return rewards

def createNetwork(learning_rate, action_size):
	input_ = tf.placeholder(tf.float32, [None, 2], name = 'input')
	actions = tf.placeholder(tf.float32, [None, action_size], name = 'actions')
	discounted_rewards = tf.placeholder(tf.float32, [None, ], name = 'discounted_rewards')
	fc1 = tf.contrib.layers.fully_connected(inputs = input_, num_outputs = 10, activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
	fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = 20, activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
	fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = action_size, activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
	action_distribution = tf.nn.softmax(fc3)
	neg_loss_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
	loss = tf.reduce_mean(neg_loss_prob * discounted_rewards)
	train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	return action_distribution, loss, train_opt


def trainPolicyNetwork():
	graph = createGraph()
	startState = startEpisode(graph)
	maximumActions = max([len(node.neighbours) for node in graph._nodes])
	learning_rate = 0.0001
	input_ = tf.placeholder(tf.float32, [None, 2], name = 'input')
	_actions = tf.placeholder(tf.float32, [None, maximumActions], name = 'actions_')
	discounted_rewards = tf.placeholder(tf.float32, [None, ], name = 'discounted_rewards')
	fc1 = tf.contrib.layers.fully_connected(inputs = input_, num_outputs = 10, activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
	fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = 20, activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
	fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = maximumActions, activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
	action_distribution = tf.nn.softmax(fc3)
	neg_loss_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = _actions)
	loss = tf.reduce_mean(neg_loss_prob * discounted_rewards)
	train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	with tf.Session() as Sess:
		Sess.run(tf.global_variables_initializer())
		for episode in range(1000000):
			state = startState
			total_reward = 0
			episode_states = []
			episode_rewards = []
			episode_actions = []
			while True:
				action_probability_distribution = Sess.run(action_distribution, feed_dict = {input_: np.array(convertState(state)).reshape(1,2)})
				actions = graph.getActions(state)
				index = int(np.random.choice(list(range(maximumActions)),1,p = action_probability_distribution.ravel()))
				action__ = np.zeros(maximumActions)
				action__[index] = 1
				episode_actions.append(action__)
				episode_states.append(convertState(state))
				try:
					action = actions[index]
					reward = graph.getReward(action, state)
					episode_rewards.append(reward)
					state = graph.getState(action, state)
				except: 
					reward = -10
					episode_rewards.append(reward)
				total_reward += reward
				try:
					if graph.terminate(action, state):
						discounted = processRewards(episode_rewards)
						_loss, _ = Sess.run([loss, train_opt], feed_dict = {input_: np.array(episode_states), _actions: np.array(episode_actions), discounted_rewards: np.array(discounted)})
						break	
				except: continue
			if episode % 10000 == 0: print('Percentage Done: ', episode/1000000, '%')
			if episode > 1000000 - 10: print(total_reward) 
	print(startState)
	print([node.neighbours for node in graph._nodes])
		


trainPolicyNetwork()