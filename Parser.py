import xml.etree.ElementTree as ET

class Parser:
    def __init__(self, filename):
        self.root = ET.parse(filename).getroot()
        self.sections = [child for child in self.root]
        self.nodes, self.links = self._getData()

    def _getData(self):
        [nodes, links] = [child for child in self.sections[0]]
        keys = []
        for child in nodes:
            keys = keys + [child.attrib['id']]
        node_data = {key: None for key in keys}
        attributes = [[axis.text for axis in coordinate] for node in nodes for coordinate in node]
        for index,key in enumerate(node_data.keys()): node_data[key] = attributes[index]
        link_data = {key: None for key in [attributes.attrib['id'] for attributes in links]}
        attributes = [[axis.text for axis in link if any(axis.tag == tag for tag in ['source', 'target'])] for link in links ]
        for index,key in enumerate(link_data.keys()): link_data[key] = attributes[index]
        return node_data, link_data

