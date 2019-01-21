from functools import total_ordering

class Graph:
    def __init__(self): pass

    def addNodes(self, nodes):
        if all(isinstance(node, Node) for node in nodes): self._nodes = nodes
        else: raise TypeError

    def _checkEdges(self, edges):
        return True if all(isinstance(edge, Edge) and edge._source and edge._destination in self._nodes for edge in edges) else False

    def _processEdges(self, edges):
        edges = edges + [edge.__rev__() for edge in edges]
        edges = [edge for index, edge in enumerate(edges) if edge not in edges[index+1:]]
        return edges

    def _linkNodes(self, edges):
        for node in self._nodes:
            node.neighbours = sorted([edge._destination for edge in self._edges if edge._source == node])
        
    def addEdges(self, edges):
        if self._checkEdges(edges):
            self._edges = self._processEdges(edges)
            self._linkNodes(edges)
        else: raise TypeError

    def sendPacket(self, packet):
        if all(node in self._nodes for node in [packet._source, packet._destinatioan]): return True
        else: return False

    def getActions(self, state): return state[0].neighbours
    def getState(self, action, state): return (action, state[1])
    def getReward(self, action, state): return -1 if action != state[1] else 1
    def terminate(self, action, state): return False if action != state[1] else True

class Packet:
    def __init__(self, source, destination):
        if all(isinstance(node, Node) for node in [source, destination]):
            self._source, self._destination = source, destination

    def __eq__(self, other): return True if self._source == other._source and self._destination == other._destination else False
    def __repr__(self): return f'Packet: {[self._source, self._destination]}'

@total_ordering
class Node:
    def __init__(self, name):
        if all(hasattr(name.__class__, attribute) for attribute in ['__lt__', '__eq__']): self.__name__ = name
        else: raise TypeError
        self.neighbours = []
    
    def __lt__(self, other): return True if self.__name__ > other.__name__ else False
    def __eq__(self, other): return True if self.__name__ == other.__name__ else False
    def __repr__(self):
        return f'Node: {self.__name__}'

class Edge:
    def __init__(self, source, destination):
        if all(isinstance(link, Node) for link in [source,destination]):
            self._source = source
            self._destination = destination
        else: raise TypeError
   
    def __eq__(self, other): return True if self._source == other._source and self._destination == other._destination else False
    def __rev__(self): return Edge(self._destination, self._source)
    def __repr__(self):
        return f'Edge: {[self._source, self._destination]}'

