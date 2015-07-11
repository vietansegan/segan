# Script to take plain text topics from SHLDA and turn them into dot file
# (suitable for further editing in OmniGraffle)
#
# Author: Jordan Boyd-Graber
# Date: 12. Feb 2013

from textwrap import TextWrapper
import sys
from math import floor
from string import strip

COLORS = ["FF8888", "F8888E", "F28894", "EC889A", "E588A1", "DF88A7",
      "D988AD", "D388B3", "CC88BA", "C688C0", "C088C6", "BA88CC",
      "B388D3", "AD88D9", "A788DF", "A188E5", "9A88EC", "9488F2",
      "8E88F8", "8888FF"]

class TopicNode:
    """
    Class to parse and display a single line of the input file.  Represents a
    node in the topic hierarchy.
    """

    def __init__(self, line):
        node_fields, line = line.split("(")
        count_fields, word_fields = line.split(")")

        self.edges = set()
        node_fields = map(strip, node_fields.split(":"))
        node_fields = ["_".join(node_fields[:(x + 1)]) for x in \
                                    xrange(len(node_fields))]
        node_fields = ['"%s"' % x for x in node_fields]

        for ii, vv in list(enumerate(node_fields))[1:]:
            self.edges.add((node_fields[ii-1], vv))
        self.id = node_fields[-1]

        count_fields = count_fields.split(";")
        self.words = map(strip, word_fields.split(" "))
        self.response = float(count_fields[-1])
        self.token_count = int(count_fields[1])
        #self.doc_count = int(count_fields[1])
        self.iter_created = int(count_fields[0])
    
        #print("Node ids: %s" % str(node_fields[-1]))
        #print("Count: %s" % str(count_fields))
	#print("Word: %s" % str(word_fields))
	#print("response = %f" % self.response)
	#print("doc count = %d" % self.doc_count)
	#print("token count = %d" % self.token_count)
	#print("iter created = %d" % self.iter_created)

    def graphviz(self, color):
        text = ""
        if len(self.words) == 2:
            text += "root"
        else:
            for ii in range(0, 10):
                text += " " + self.words[ii]
        text += "\n(" + str(self.id).replace('\"', '') + ", " + str(self.response) + ", " + str(self.token_count) + ")"
        return '\t%s [color="#%s" label="%s"];' % \
            (self.id, color, text)


class TopicGraph:
    """
    Class to represent the entire topic hierarchy and to keep track of
    statistics necessary to render it.
    """

    def __init__(self, doc_lim):
        self._edges = set()
        self._doc_lim = doc_lim
        self._nodes = {}
        self._max_response = float("-inf")
        self._min_response = float("inf")

    def range(self):
        return self._min_response, self._max_response

    def add_node(self, line, reverse=False):
        node = TopicNode(line)

        #if node.doc_count < self._doc_lim:
        #    return None
        
        if node.iter_created > 100:
            return None

        if reverse:
            node.response *= -1.0

        self._nodes[node.id] = node
        for ii in node.edges:
            self._edges.add(ii)

        self._max_response = max(self._max_response, node.response)
        self._min_response = min(self._min_response, node.response)

        return node

    def color(self, response, min, max):
        # Add a little bit of padding so that the max value doesn't break
        # things
        width = (max - min) * 1.001

        levels = len(COLORS)

        val = int(floor((response - min) / width * levels))

        return COLORS[val]

    def graphviz(self):
        yield 'graph {'
        yield '\tnode [style="filled" shape=box width=2.5in height=1.5in fixedsize=true];'

        for nn in self._nodes:
            node = self._nodes[nn]
            color = self.color(node.response, self._min_response,
                               self._max_response)

            yield node.graphviz(color)

        yield ""

        for ee in self._edges:
            yield '\t%s -- %s;' % ee
        yield '}'


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: dot_topics.py INFILE OUTFILE [DOC_LIM] [--reverse]")

    infile = sys.argv[1]
    outfile = sys.argv[2]

    doc_lim = 15
    if len(sys.argv) > 3:
        doc_lim = int(sys.argv[3])

    reverse = False
    if len(sys.argv) > 4 and sys.argv[4] == "--reverse":
        print("Reverse")
        reverse = True

    graph = TopicGraph(doc_lim)
    for ii in [x for x in open(infile) if x.strip()]:
        graph.add_node(ii, reverse)

    print("Range: %s" % str(graph.range()))

    o = open(outfile, 'w')
    for ii in graph.graphviz():
        o.write("%s\n" % ii)

    o.close()
