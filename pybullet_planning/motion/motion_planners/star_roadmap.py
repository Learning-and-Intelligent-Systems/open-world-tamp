from collections import Mapping

class StarRoadmap(Mapping, object):

    def __init__(self, center, planner_fn):
        self.center = center # TODO: plan instead from the closest point on the roadmap
        self.planner_fn = planner_fn
        self.roadmap = {}

    """
    def __getitem__(self, q):
        return self.roadmap[q]

    def __len__(self):
        return len(self.roadmap)

    def __iter__(self):
        return iter(self.roadmap)
    """

    def grow(self, goal):
        if goal not in self.roadmap:
            self.roadmap[goal] = self.planner_fn(self.center, goal)
        return self.roadmap[goal]

    def __call__(self, start, goal):
        start_traj = self.grow(start)
        if start_traj is None:
            return None
        goal_traj = self.grow(goal)
        if goal_traj is None:
            return None
        return start_traj.reverse(), goal_traj
