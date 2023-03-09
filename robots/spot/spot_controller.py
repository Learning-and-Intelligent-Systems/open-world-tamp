from open_world.simulation.controller import Controller

class SpotController(Controller):
    def __init__(self, robot, verbose=True, **kwargs):
        self.robot = robot
        super(SpotController, self).__init__(self.robot)