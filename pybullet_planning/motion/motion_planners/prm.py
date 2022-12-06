from collections import namedtuple, Mapping
from heapq import heappop, heappush
import operator
import time

from .utils import INF, get_pairs, merge_dicts, flatten, RED, apply_alpha, default_selector


# TODO - Visibility-PRM, PRM*

class Vertex(object):

    def __init__(self, q):
        self.q = q
        self.edges = {}
        self._handle = None

    def clear(self):
        self._handle = None

    def draw(self, env, color=apply_alpha(RED, alpha=0.5)):
        # https://github.mit.edu/caelan/lis-openrave
        from manipulation.primitives.display import draw_node
        self._handle = draw_node(env, self.q, color=color)

    def __str__(self):
        return 'Vertex(' + str(self.q) + ')'
    __repr__ = __str__


class Edge(object):

    def __init__(self, v1, v2, path):
        self.v1, self.v2 = v1, v2
        self.v1.edges[v2], self.v2.edges[v1] = self, self
        self._path = path
        #self._handle = None
        self._handles = []

    def end(self, start):
        if self.v1 == start:
            return self.v2
        if self.v2 == start:
            return self.v1
        assert False

    def path(self, start):
        if self._path is None:
            return [self.end(start).q]
        if self.v1 == start:
            return self._path + [self.v2.q]
        if self.v2 == start:
            return self._path[::-1] + [self.v1.q]
        assert False

    def configs(self):
        if self._path is None:
            return []
        return [self.v1.q] + self._path + [self.v2.q]

    def clear(self):
        #self._handle = None
        self._handles = []

    def draw(self, env, color=apply_alpha(RED, alpha=0.5)):
        if self._path is None:
            return
        # https://github.mit.edu/caelan/lis-openrave
        from manipulation.primitives.display import draw_edge
        #self._handle = draw_edge(env, self.v1.q, self.v2.q, color=color)
        for q1, q2 in get_pairs(self.configs()):
            self._handles.append(draw_edge(env, q1, q2, color=color))

    def __str__(self):
        return 'Edge(' + str(self.v1.q) + ' - ' + str(self.v2.q) + ')'
    __repr__ = __str__

##################################################

SearchNode = namedtuple('SearchNode', ['cost', 'parent'])


class Roadmap(Mapping, object):

    def __init__(self, samples=[]):
        self.vertices = {}
        self.edges = []
        self.add(samples)

    def __getitem__(self, q):
        return self.vertices[q]

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def __call__(self, q1, q2):
        if q1 not in self or q2 not in self:
            return None
        start, goal = self[q1], self[q2]
        queue = [(0, start)]
        nodes, processed = {start: SearchNode(0, None)}, set()

        def retrace(v):
            pv = nodes[v].parent
            if pv is None:
                return [v.q]
            return retrace(pv) + v.edges[pv].path(pv)

        while len(queue) != 0:
            _, cv = heappop(queue)
            if cv in processed:
                continue
            processed.add(cv)
            if cv == goal:
                return retrace(cv)
            for nv, edge in cv.edges.items():
                cost = nodes[cv].cost + len(edge.path(cv))
                if nv not in nodes or cost < nodes[nv].cost:
                    nodes[nv] = SearchNode(cost, cv)
                    heappush(queue, (cost, nv))
        return None

    def add(self, samples):
        new_vertices = []
        for q in samples:
            if q not in self:
                self.vertices[q] = Vertex(q)
                new_vertices.append(self[q])
        return new_vertices

    def connect(self, v1, v2, path=None):
        if v1 not in v2.edges:  # TODO - what about parallel edges?
            edge = Edge(v1, v2, path)
            self.edges.append(edge)
            return edge
        return None

    def clear(self):
        for v in self.vertices.values():
            v.clear()
        for e in self.edges:
            e.clear()

    def draw(self, env):
        for v in self.vertices.values():
            v.draw(env)
        for e in self.edges:
            e.draw(env)

    @staticmethod
    def merge(*roadmaps):
        new_roadmap = Roadmap()
        new_roadmap.vertices = merge_dicts(
            *[roadmap.vertices for roadmap in roadmaps])
        new_roadmap.edges = list(
            flatten(roadmap.edges for roadmap in roadmaps))
        return new_roadmap


class PRM(Roadmap):

    def __init__(self, distance_fn, extend_fn, collision_fn, samples=[]):
        super(PRM, self).__init__()
        self.distance_fn = distance_fn
        self.extend_fn = extend_fn
        self.collision_fn = collision_fn
        self.grow(samples)

    def grow(self, samples):
        raise NotImplementedError()

    def __call__(self, q1, q2):
        self.grow(samples=[q1, q2])
        if q1 not in self or q2 not in self:
            return None
        start, goal = self[q1], self[q2]
        heuristic = lambda v: self.distance_fn(v.q, goal.q)  # lambda v: 0

        queue = [(heuristic(start), start)]
        nodes, processed = {start: SearchNode(0, None)}, set()

        def retrace(v):
            if nodes[v].parent is None:
                return [v.q]
            return retrace(nodes[v].parent) + v.edges[nodes[v].parent].path(nodes[v].parent)

        while len(queue) != 0:
            _, cv = heappop(queue)
            if cv in processed:
                continue
            processed.add(cv)
            if cv == goal:
                return retrace(cv)
            for nv in cv.edges:
                cost = nodes[cv].cost + self.distance_fn(cv.q, nv.q)
                if (nv not in nodes) or (cost < nodes[nv].cost):
                    nodes[nv] = SearchNode(cost, cv)
                    heappush(queue, (cost + heuristic(nv), nv))
        return None

##################################################

class DistancePRM(PRM):

    def __init__(self, distance_fn, extend_fn, collision_fn, samples=[], connect_distance=.5):
        self.connect_distance = connect_distance
        super(self.__class__, self).__init__(
            distance_fn, extend_fn, collision_fn, samples=samples)

    def grow(self, samples):
        old_vertices = self.vertices.keys()
        new_vertices = self.add(samples)
        for i, v1 in enumerate(new_vertices):
            for v2 in new_vertices[i + 1:] + old_vertices:
                if self.distance_fn(v1.q, v2.q) <= self.connect_distance:
                    path = list(self.extend_fn(v1.q, v2.q))[:-1]
                    if not any(self.collision_fn(q) for q in default_selector(path)):
                        self.connect(v1, v2, path)
        return new_vertices


class DegreePRM(PRM):

    def __init__(self, distance_fn, extend_fn, collision_fn, samples=[], target_degree=4, connect_distance=INF):
        self.target_degree = target_degree
        self.connect_distance = connect_distance
        super(self.__class__, self).__init__(
            distance_fn, extend_fn, collision_fn, samples=samples)

    def grow(self, samples):
        # TODO: do sorted edges version
        new_vertices = self.add(samples)
        if self.target_degree == 0:
            return new_vertices
        for v1 in new_vertices:
            degree = 0
            for _, v2 in sorted(filter(lambda pair: (pair[1] != v1) and (pair[0] <= self.connect_distance),
                                       map(lambda v: (self.distance_fn(v1.q, v.q), v), self.vertices.values())),
                                key=operator.itemgetter(0)): # TODO - slow, use nearest neighbors
                if self.target_degree <= degree:
                    break
                if v2 not in v1.edges:
                    path = list(self.extend_fn(v1.q, v2.q))[:-1]
                    if not any(self.collision_fn(q) for q in default_selector(path)):
                        self.connect(v1, v2, path)
                        degree += 1
                else:
                    degree += 1
        return new_vertices

##################################################

def prm(start, goal, distance_fn, sample_fn, extend_fn, collision_fn,
        target_degree=4, connect_distance=INF, num_samples=100): #, max_time=INF):
    """
    :param start: Start configuration - conf
    :param goal: End configuration - conf
    :param distance_fn: Distance function - distance_fn(q1, q2)->float
    :param sample_fn: Sample function - sample_fn()->conf
    :param extend_fn: Extension function - extend_fn(q1, q2)->[q', ..., q"]
    :param collision_fn: Collision function - collision_fn(q)->bool
    :return: Path [q', ..., q"] or None if unable to find a solution
    """
    # TODO: compute_graph
    start_time = time.time()
    start = tuple(start)
    goal = tuple(goal)
    samples = [start, goal] + [tuple(sample_fn()) for _ in range(num_samples)]
    if target_degree is None:
        roadmap = DistancePRM(distance_fn, extend_fn, collision_fn, samples=samples,
                              connect_distance=connect_distance)
    else:
        roadmap = DegreePRM(distance_fn, extend_fn, collision_fn, samples=samples,
                            target_degree=target_degree, connect_distance=connect_distance)
    return roadmap(start, goal)
