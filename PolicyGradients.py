from Architechture import Node, Edge, Graph
from Parser import Parser
from random import sample, seed, randint
import numpy as np
import tensorflow as tf
from time import sleep

seed(22)
np.random.seed(22)

def createGraph():
    parser = Parser('germany50.xml')
    nodes, links = parser.nodes, parser.links
    node_names = list(nodes.keys())
    graph = Graph()
    graph.addNodes([Node(node) for node in range(len(node_names))])
    nodes = graph._nodes
    edges = [Edge(nodes[source], nodes[destination]) for [source, destination] in [[node_names.index(link[0]), node_names.index(link[1])] for _,link in links.items()]]
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
	fc1 = tf.contrib.layers.fully_connected(inputs = input_, num_outputs = 512, activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
	fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = 1024, activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
	fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = action_size, activation_fn = tf.nn.relu, weights_initializer = tf.contrib.layers.xavier_initializer())
	action_distribution = tf.nn.softmax(fc3)
	neg_loss_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
	loss = tf.reduce_mean(neg_loss_prob * discounted_rewards)
	train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	return action_distribution, loss, train_opt


def trainPolicyNetwork():
    graph = createGraph()
    startState = startEpisode(graph)
    print(startState)
    maximumActions = max([len(node.neighbours) for node in graph._nodes])
    print(maximumActions)
    learning_rate = 0.0001
    input_ = tf.placeholder(tf.float32, [None, 2], name = 'input')
    _actions = tf.placeholder(tf.float32, [None, maximumActions], name = 'actions_')
    discounted_rewards = tf.placeholder(tf.float32, [None, ], name = 'discounted_rewards')
    fc1 = tf.contrib.layers.fully_connected(inputs = input_, num_outputs = 512, activation_fn = tf.nn.relu, weights_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01))
    fc2 = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = 1024, activation_fn = tf.nn.relu, weights_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01))
    fc3 = tf.contrib.layers.fully_connected(inputs = fc2, num_outputs = maximumActions, activation_fn = tf.nn.relu, weights_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01))
    action_distribution = tf.nn.softmax(fc3)
    neg_loss_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = _actions)
    loss = tf.reduce_mean(neg_loss_prob * discounted_rewards)
    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    epoch_reward = 0
    epoch_rewards = []
    with tf.Session() as Sess:
        Sess.run(tf.global_variables_initializer())
        for episode in range(1000000):
            state = startState
            total_reward = 0
            episode_states = []
            episode_rewards = []
            episode_actions = []
            accumulator = 0
            while True:
                accumulator += 1
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
                epoch_reward += reward
                try:
                        if graph.terminate(action, state) or accumulator > 30:
                                discounted = processRewards(episode_rewards)
                                _loss, _ = Sess.run([loss, train_opt], feed_dict = {input_: np.array(episode_states), _actions: np.array(episode_actions), discounted_rewards: np.array(discounted)})
                                break	
                except: pass
            if episode % 10000 == 0 and episode != 0: 
                epoch_rewards.append(epoch_reward)
                print('Percentage Done: ', episode/1000000, '%')
                print('Epoch Reward: ', epoch_reward)
                epoch_reward = 0
            if all(0.98*max(epoch_rewards) < x for x in epoch_rewards[-5:]) and len(epoch_rewards) > 5: 
                print('Maximum Reward Consistently Reached')
                break

        state = startState
        #while True:
        #    action_probability_distribution = Sess.run(action_distribution, feed_dict = {input_: np.array(convertState(state)).reshape(1,2)})
        #    actions = graph.getActions(state)
        #    index = int(np.random.choice(list(range(maximumActions)),1,p = action_probability_distribution.ravel()))
        #    action = actions[index]
        #    print('State: ', state)
        #    print('Actions: ', actions)
        #    print('Actions Distros:', action_probability_distribution)
        #    state = graph.getState(action, state)
        #    if graph.terminate(action, state): 
        #        print('Done')
        #        break

                            
trainPolicyNetwork()
