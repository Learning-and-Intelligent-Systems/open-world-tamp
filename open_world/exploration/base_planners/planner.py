from abc import ABC, abstractmethod
class Planner(ABC):

    @abstractmethod
    def get_plan(self, **kwargs):
        pass
