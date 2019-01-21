from Architechture import Graph, Node, Edge
from random import sample, seed, randint
from time import sleep
import numpy as np

seed(60)
np.random.seed(60)

def createGraph():
    graph = Graph()
    graph.addNodes([Node(index) for index in range(10)])
    nodes = graph._nodes
    graph.addEdges([Edge(nodes[source], nodes[destination]) for [source, destination] in [sample(range(10), 2) for _ in range(40)]]) 
    return graph
    
def startEpisode(graph): return tuple(sample(graph._nodes, 2))

def constructData(graph):
    episodes = range(100)
    memory = {'States': [], 'Rewards': []}
    for episode in episodes:
        state = startEpisode(graph)
        for reward,state in simulate(graph, state):
            try: 
                memory['States'][episode] = memory['States'][episode] + [state]
                memory['Rewards'][episode] = memory['Rewards'][episode] + [reward]
            except: 
                memory['States'] = memory['States'] + [[state]]
                memory['Rewards'] = memory['Rewards'] + [[reward]]

    memory = discountRewards(memory)
    memory = {key: (lambda x: [value for data in x for value in data])(data) for key, data in memory.items()}
    memory['States'] = list(map(convertState, memory['States']))
    return memory['States'], memory['Rewards']

def convertState(state): return [int(state[0].__name__), int(state[1].__name__)]
def discountRewards(memory, gamma = 0.80): return {'States': memory['States'], 'Rewards': list(map(discount, memory['Rewards']))}

def discount(rewards, gamma = 0.80):
    for iteration in range(rewards.__len__()):
        rewards[iteration] = sum(list(map(lambda x: x[1]*(gamma**x[0]), enumerate(rewards[iteration:]))))
    return rewards
    
def simulate(graph, state):
    while True:
        actions = graph.getActions(state)
        [action] = sample(actions, 1)
        reward = graph.getReward(action, state)
        state = graph.getState(action, state)
        yield reward, state
        if graph.terminate(action, state): break

def sigmoid(x, derivative = False):
    if derivative: 
        return sigmoid(x)*sigmoid(1-x)
    return 1/(1 + np.exp(-x))

def trainPolicyNetwork():
    graph = createGraph()
    stateStart = startEpisode(graph)
    maximumActions = max([len(node.neighbours) for node in graph._nodes])
    weights = {'Weights0': 2*np.random.random((2,5))-1, 'Weights1': 2*np.random.random((5,maximumActions))-1}
    for _ in range(10000000):
        state = stateStart
        data = {'Actions': [], 'Rewards': [], 'States': []}
        while True:
            layer1 = sigmoid(np.dot(convertState(state), weights['Weights0']))
            layer2 = sigmoid(np.dot(layer1, weights['Weights1']))
            layer2 = layer2/sum(layer2)
            data['States'] = data['States'] + [state]
            try:
                index = int(np.random.choice(list(range(maximumActions)),1,p = layer2))
                data['Actions'] = data['Actions'] + [index]
                action = graph.getActions(state)[index]
                reward = graph.getReward(action, state)
                data['Rewards'] = data['Rewards'] + [reward]
                state  = graph.getState(action, state)
                if graph.terminate(action, state): break
            except: 
                reward = -10
                data['Rewards'] = data['Rewards'] + [reward]

        data['Rewards'] = discount(data['Rewards'])
        loss = calculateLoss(data, maximumActions)
        for index in range(len(loss)):
            layer1 = sigmoid(np.dot(convertState(state), weights['Weights0']))
            layer2 = sigmoid(np.dot(layer1, weights['Weights1']))
            delta2 = loss[index] * sigmoid(layer2, derivative = True)
            delta1 = delta2.dot(weights['Weights1'].T) * sigmoid(layer1, derivative = True)
            weights['Weights1'] += layer1.reshape(1,5).T.dot(delta2.reshape(1,maximumActions))*0.001
            weights['Weights0'] += np.array(convertState(data['States'][index])).reshape(1,2).T.dot(delta1.reshape(1,5))*0.05
    print(sum(data['Rewards']))

def calculateLoss(data, maximumActions):
    loss = np.zeros((len(data['Rewards']), maximumActions))
    for index in range(len(loss)):
        loss[index][data['Actions'][index]] = data['Rewards'][index]
    maximumLoss = np.max(np.abs(loss.flatten())) + 1
    for column, row in np.ndindex(loss.shape):
        if loss[column][row] != 0: loss[column][row] = (loss[column][row] - 1)/maximumLoss
    return loss

    

trainPolicyNetwork()













