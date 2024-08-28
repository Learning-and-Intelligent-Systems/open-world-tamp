import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
from matplotlib.patches import Rectangle

from owt.pb_utils import get_aabb

GRID_RESOLUTION = 0.2  # Grid resolutions


class Graph:
    """Graph class that discretices the configuration space."""

    def __init__(self):
        self.startpos = None
        self.endpos = None

        # Holds all the available vertices
        self.vertices = []
        # Holds all the edges in the graph
        self.edges = []
        # Whether the graph connects the start and the end node
        self.success = False

        # A dictionary mapping vertices to their index in the graph
        self.vex2idx = dict()
        # A dictionary that contains the neighbors of every vertex.
        self.neighbors = dict()
        # A dictionary containing the distance of every edge.
        self.distances = dict()
        # A dictionary containing neighbors from only rotating
        self.rot_neighbors = dict()

        # The step of the angle theta as discretized by the graph
        self.t_step = None

        # Holds a pyplot figure for dynamic plotting.
        self.fig = None

    def add_vex(self, pos):
        """Adds a vertex to the graph.

        Args:
            pos (tuple): position of the vertex to add.
        Returns:
            int: index of the vertex inside the graph.
        """
        try:
            idx = self.vex2idx[pos]
        except:
            idx = len(self.vertices)
            self.vertices.append(pos)
            self.vex2idx[pos] = idx
            self.neighbors[idx] = []
            self.rot_neighbors[idx] = []
        return idx

    def dettach_vex(self, pos):
        """Dettach a vertex from all of its neighbors, eliminating all the
        edges to and from it.

        Args:
            pos (tuple): position of the vertex to add.
        """
        idx = self.vex2idx[pos]
        self.vex2idx[pos] = None
        for idx2 in self.neighbors[idx]:
            if (idx, idx2) in self.edges:
                self.edges.remove((idx, idx2))
            else:
                self.edges.remove((idx2, idx))
            self.neighbors[idx2].remove(idx)
        self.neighbors[idx] = None

    def get_vertex_rot(self, vex, direction):
        """Returns a neeighbor to a vertex from a rotation.

        Args:
            vex (tuple): The vertex to get its neighbor from
            direction(int): Either -1 or 1 depending on the direction. 1 for counterclockwise
                            -1 for clockwise
        Returns:
            The vertex after the executed rotation
        """
        new_rot = vex[2] + direction * self.res[2]
        new_rot = new_rot + 2 * np.pi if new_rot < 0 else new_rot
        new_rot = (
            new_rot - 2 * np.pi if round(new_rot, 2) >= round(2 * np.pi, 2) else new_rot
        )

        rot_1, rot_2 = (self.vertices[x] for x in self.rot_neighbors[self.vex2idx[vex]])
        return min(rot_1, rot_2, key=lambda x: abs(x[2] - new_rot))

    def add_edge(self, idx1, idx2):
        """Adds an edge between two vertices.

        Args:
            idx1 (int): index of the first vertex
            idx2 (int): index of the second vertex
        """
        self.edges.append((idx1, idx2))
        self.neighbors[idx1].append(idx2)
        self.neighbors[idx2].append(idx1)
        if self.vertices[idx1][:2] == self.vertices[idx2][:2]:
            self.rot_neighbors[idx1].append(idx2)
            self.rot_neighbors[idx2].append(idx1)

    def initialize_full_graph(self, env, resolution):
        """Given an environment and a desired resolution create a 6-connected
        graph.

        Args:
            env (object): The working environment to represent.
            resolution (list): A list defining the resolution along the x axis, y axis, and the angle.
        """
        # Gets how many nodes along each dimension.
        self.x_step = (
            int((env.room.aabb.upper[0] - env.room.aabb.lower[0]) / resolution[0]) + 1
        )
        self.y_step = (
            int((env.room.aabb.upper[1] - env.room.aabb.lower[1]) / resolution[1]) + 1
        )
        self.t_step = int((2 * np.pi) / (resolution[2])) if resolution[2] != 0 else 1

        # Saves resolution for later access.
        self.res = resolution

        # Creates all the nodes.
        x = env.room.aabb.lower[0] - resolution[0]
        for i in range(self.x_step):
            y = env.room.aabb.lower[1] - resolution[1]
            x += resolution[0]
            for j in range(self.y_step):
                y += resolution[1]
                t = -resolution[2]
                for k in range(self.t_step):
                    t += resolution[2]
                    self.add_vex((round(x, 2), round(y, 2), round(t, 3)))

        # Adds the edges to make the graph 6-connected. Moving horizontally and vertically as well
        # as turning a certain angle to both sides.
        n_vertices = self.x_step * self.y_step * self.t_step
        for i in range(n_vertices):
            # Add turn edges
            if (i + 1) % self.t_step != 0:
                self.add_edge(i, i + 1)
            else:
                self.add_edge(i, i + 1 - self.t_step)

            # Add movement edges
            if i % (self.y_step * self.t_step) < ((self.y_step - 1) * self.t_step):
                self.add_edge(i, i + self.t_step)

            if i < (n_vertices - (self.t_step * self.y_step)):
                self.add_edge(i, i + self.y_step * self.t_step)

    def rand_vex(self, env):
        """Selects a random vertex from the graph.

        Args:
            env (object): The environment where the graph is constructed.
        Returns:
            tuple: a vertex of the graph.
        """
        while True:
            i = np.random.randint(len(self.vertices))
            # Restrict the random points to be on the floors' bounds to improve sampling in
            # non-rectangular rooms
            for floor in env.room.floors:
                aabb = get_aabb(floor)
                if np.all(
                    [l1 <= l2 for l1, l2 in zip(aabb.lower[:2], self.vertices[i][:2])]
                ) & np.all(
                    [l2 <= l1 for l1, l2 in zip(aabb.upper[:2], self.vertices[i][:2])]
                ):
                    return self.vertices[i]

    def plot(self, env, path=None):
        """Visualizes the constructed graph in the environment.

        Args:
            env (object): The environment where the graph is constructed.
            path (list): A path in the graph to highlight.
        """
        px, py, _ = zip(*self.vertices)
        fig, ax = plt.subplots()

        # Scatters the nodes of the graph.
        ax.scatter(px, py, c="cyan")
        if self.startpos != None and self.endpos != None:
            ax.scatter(self.startpos[0], self.startpos[1], c="black")
            ax.scatter(self.endpos[0], self.endpos[1], c="red")

        lines = [
            (self.vertices[edge[0]][0:2], self.vertices[edge[1]][0:2])
            for edge in self.edges
        ]
        lc = mc.LineCollection(lines, colors="green", linewidths=2)
        ax.add_collection(lc)

        # Draw angles of points
        angle_lines = []
        for x, y, t in self.vertices:
            endy = y + 0.05 * np.sin(t)
            endx = x + 0.05 * np.cos(t)
            angle_lines.append(((x, y), (endx, endy)))
        lc = mc.LineCollection(angle_lines, colors="red", linewidths=2)
        ax.add_collection(lc)

        # Draw room shape
        for wall in env.room.walls:
            wall_aabb = get_aabb(wall)
            rec = Rectangle(
                (wall_aabb.lower[0:2]),
                wall_aabb.upper[0] - wall_aabb.lower[0],
                wall_aabb.upper[1] - wall_aabb.lower[1],
                color="grey",
                linewidth=0.1,
            )
            ax.add_patch(rec)

        # Draws the obstacles aabb in the room.
        # TODO: Not taking rotations into account. Change later for better visualization
        for obstacle in env.static_objects + env.movable_boxes:
            color = "brown"
            if isinstance(obstacle, int):
                aabb = get_aabb(obstacle)
            else:
                aabb = obstacle.aabb
                color = "yellow"
            ax.add_patch(
                Rectangle(
                    (aabb.lower[0], aabb.lower[1]),
                    aabb.upper[0] - aabb.lower[0],
                    aabb.upper[1] - aabb.lower[1],
                    color=color,
                    linewidth=0.1,
                )
            )

        # If a pth was given, highlight it on the graph
        if path is not None:
            paths = [(path[i][0:2], path[i + 1][0:2]) for i in range(len(path) - 1)]
            lc2 = mc.LineCollection(paths, colors="blue", linewidths=3)
            ax.add_collection(lc2)

        ax.autoscale()
        ax.margins(0.1)
        plt.show()

    def plot_search_dynamic(self, env, node):
        """Dynamically visualize the search as it is being executed.

        Args:
            env (object): The environment where the graph is constructed.
            node (tuple): New node of the graph to visualize
        """
        if self.fig == None:
            self.fig, self.ax = plt.subplots()
            # Draw room shape
            for wall in env.room.walls:
                wall_aabb = get_aabb(wall)
                rec = Rectangle(
                    (wall_aabb.lower[0:2]),
                    wall_aabb.upper[0] - wall_aabb.lower[0],
                    wall_aabb.upper[1] - wall_aabb.lower[1],
                    color="grey",
                    linewidth=0.1,
                )
                self.ax.add_patch(rec)

            # Draws the obstacles aabb in the room.
            # TODO: Not taking rotations into account. Change later for better visualization
            for obstacle in env.static_objects + env.movable_boxes:
                color = "brown"
                if isinstance(obstacle, int):
                    aabb = get_aabb(obstacle)
                else:
                    aabb = obstacle.aabb
                    color = "yellow"
                self.ax.add_patch(
                    Rectangle(
                        (aabb.lower[0], aabb.lower[1]),
                        aabb.upper[0] - aabb.lower[0],
                        aabb.upper[1] - aabb.lower[1],
                        color=color,
                        linewidth=0.1,
                    )
                )

        # Adds the node into the visualization
        x = [node[0]]
        y = [node[1]]
        t = [node[2]]

        self.ax.scatter(x, y, c="cyan")

        # Adds the angle of the node as a red marker indicating direction.
        angle_lines = []
        endy = y[0] + 0.05 * np.sin(t[0])
        endx = x[0] + 0.05 * np.cos(t[0])
        angle_lines.append(((x[0], y[0]), (endx, endy)))
        lc = mc.LineCollection(angle_lines, colors="red", linewidths=2)
        self.ax.add_collection(lc)

        self.ax.autoscale()
        self.ax.margins(0.1)
        plt.show(block=False)
        plt.pause(0.05)

    def plot_search(
        self, env, extended, path=None, R=set(), goal=None, enforced_obstacles=[]
    ):
        """Visualizes the finished search by visualizing all the nodes it
        extended.

        Args:
            env (object): The environment where the graph is constructed.
            extended (set): A set of nodes that were extended by the search
            path (list): A path in the graph to highlight.
            R (set): Area of interest to visualize
        """
        fig, ax = plt.subplots()
        # Draw room shape
        for wall in env.room.walls:
            wall_aabb = get_aabb(wall)
            rec = Rectangle(
                (wall_aabb.lower[0:2]),
                wall_aabb.upper[0] - wall_aabb.lower[0],
                wall_aabb.upper[1] - wall_aabb.lower[1],
                color="grey",
                linewidth=0.1,
            )
            ax.add_patch(rec)

        # Draws the obstacles aabb in the room.
        # TODO: Not taking rotations into account. Change later for better visualization
        for obstacle in env.static_objects + env.movable_boxes:
            color = "brown"
            if isinstance(obstacle, int):
                aabb = get_aabb(obstacle)
            else:
                aabb = obstacle.aabb
                color = "yellow"
            ax.add_patch(
                Rectangle(
                    (aabb.lower[0], aabb.lower[1]),
                    aabb.upper[0] - aabb.lower[0],
                    aabb.upper[1] - aabb.lower[1],
                    color=color,
                    linewidth=0.1,
                )
            )

        for obstacle in enforced_obstacles:
            color = "blue"
            aabb = obstacle.aabb
            ax.add_patch(
                Rectangle(
                    (aabb.lower[0], aabb.lower[1]),
                    aabb.upper[0] - aabb.lower[0],
                    aabb.upper[1] - aabb.lower[1],
                    color=color,
                    linewidth=0.1,
                )
            )

        # Scatters all the points extended by the search.
        px, py, _ = zip(*extended)
        ax.scatter(px, py, c="cyan")

        angle_lines = []
        for x, y, t in extended:
            endy = y + 0.05 * np.sin(t)
            endx = x + 0.05 * np.cos(t)
            angle_lines.append(((x, y), (endx, endy)))
        lc = mc.LineCollection(angle_lines, colors="red", linewidths=2)
        ax.add_collection(lc)

        if path is not None:
            paths = [(path[i][0:2], path[i + 1][0:2]) for i in range(len(path) - 1)]
            lc2 = mc.LineCollection(paths, colors="blue", linewidths=3)
            ax.add_collection(lc2)

        if R is not None:
            px = [p[0] * GRID_RESOLUTION for p in R]
            py = [p[1] * GRID_RESOLUTION for p in R]
            ax.scatter(px, py, c="black", marker="s")

        if goal is not None:
            ax.scatter(goal[0], goal[1], c="black")

        ax.autoscale()
        ax.margins(0.1)
        plt.show()


def distance(vex1, vex2):
    """Helper function that returns the Euclidean distance between two tuples
    of size 2.

    Args:
        vex1 (tuple): The first tuple
        vex2 (tuple): The second tuple
    Returns:
        float: The Euclidean distance between both tuples.
    """
    return ((vex1[0] - vex2[0]) ** 2 + (vex1[1] - vex2[1]) ** 2) ** 0.5
