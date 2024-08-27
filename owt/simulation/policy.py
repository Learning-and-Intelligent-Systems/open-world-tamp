from __future__ import print_function

import datetime
import time
import warnings

warnings.filterwarnings("ignore")  # , category=DeprecationWarning)

import os

import numpy as np
import pybullet as p
from open_world.estimation.belief import GRASP_EXPERIMENT, Belief
from open_world.estimation.dnn import init_sc, init_seg
from open_world.estimation.geometry import cloud_from_depth
from open_world.estimation.observation import save_camera_images
from open_world.estimation.tables import estimate_surfaces
from open_world.exploration.environment import Environment
from open_world.exploration.utils import GRID_RESOLUTION
from open_world.planning.planner import (iterate_sequence, plan_pddlstream,
                                         post_process)
from open_world.planning.primitives import WorldState
from open_world.simulation.entities import get_label_counts
from open_world.simulation.tasks import Task
from pybullet_planning.pybullet_tools.utils import (INF, SEPARATOR,
                                                    CameraImage, LockRenderer,
                                                    WorldSaver, elapsed_time,
                                                    get_link_pose, get_pose,
                                                    invert, irange,
                                                    joint_from_name,
                                                    link_from_name, multiply,
                                                    set_joint_positions,
                                                    wait_if_gui, write_pickle)

SUCCESS_STATUS = True
FAILURE_STATUS = False
ONGOING_STATUS = None


def reset(args):
    return Task(
        goal_parts=[], arms=args.arms, skills=[], return_init=True, empty_arms=True
    )


def seg_from_gt(seg_image):
    new_seg_image = []
    for r in range(seg_image.shape[0]):
        new_seg_row = []
        for c in range(seg_image.shape[1]):
            new_seg_row.append(["other", "instance_{}".format(int(seg_image[r, c, 0]))])
        new_seg_image.append(new_seg_row)

    return np.array(new_seg_image)


def link_seg_from_gt(seg_image):
    new_seg_image = []
    for r in range(seg_image.shape[0]):
        new_seg_row = []
        for c in range(seg_image.shape[1]):
            new_seg_row.append(
                [
                    "other",
                    "instance_{}_{}".format(
                        int(seg_image[r, c, 0]), int(seg_image[r, c, 1])
                    ),
                ]
            )
        new_seg_image.append(new_seg_row)

    return np.array(new_seg_image)


def fuse_predicted_labels(
    seg_network,
    camera_image,
    fuse=False,
    use_depth=False,
    debug=False,
    num_segs=1,
    **kwargs
):
    rgb, depth, bullet_seg, _, camera_matrix = camera_image
    if fuse:
        print("Ground truth:", get_label_counts(bullet_seg))

    point_cloud = None
    if use_depth:
        point_cloud = cloud_from_depth(camera_matrix, depth)

    predicted_seg = seg_network.get_seg(
        rgb[:, :, :3],
        point_cloud=point_cloud,
        depth_image=depth,
        return_int=False,
        num_segs=num_segs,
        **kwargs
    )
    return CameraImage(rgb, depth, predicted_seg, *camera_image[3:])


class Policy(object):
    def __init__(self, args, robot, known=[], teleport=False, client=None, **kwargs):

        self.args = args
        self.robot = robot
        self.known = tuple(known)
        self.executed = False
        self.teleport = teleport
        self.client = client

        self.data = []
        self.runtimes = {}
        self.estimates = []
        self.renders = []
        self.plans = []
        self.executions = []

        # TODO: could store surfaces, objects, etc.
        self.data_folder = "run_data"

        if not os.path.exists(self.data_folder):
            # Create a new directory because it does not exist
            os.makedirs(self.data_folder)

        self.robot.update_conf()
        if args.segmentation:
            self.seg_network = init_seg(
                branch=self.args.segmentation_model,
                maskrcnn_rgbd=args.maskrcnn_rgbd,
                post_classifier=args.fasterrcnn_detection,
            )

        self.sc_network = None
        if args.shape_completion:
            self.sc_network = init_sc(branch=args.shape_completion_model)

        self.belief = Belief(
            self.robot,
            surface_beliefs=[
                # SurfaceBelief(table, resolutions=0.04 * np.ones(3), known_objects=real_world.known),
            ],
            client=self.client,
        )

    ##################################################
    def open_grippers(self, blocking=False, timeout=5.0):
        clients = [
            self.robot.controller.open_gripper(arm, blocking=blocking)
            for arm in self.robot.arms
        ]
        return self.robot.controller.wait_for_clients(clients, timeout=timeout)

    def reset_belief(self):
        self.belief.reset()
        for surface in self.belief.known_surfaces:
            surface.remove()
        self.belief.known_objects = self.belief.known_surfaces = []
        return self.belief

    ##################################################

    def get_image(self):

        if not self.args.real_camera:
            [camera] = self.robot.cameras
            camera_image = camera.get_image()  # TODO: remove_alpha

            if self.args.segmentation:
                camera_image = fuse_predicted_labels(
                    self.seg_network,
                    camera_image,
                    use_depth=self.args.segmentation_model != "maskrcnn",
                )
                save_camera_images(camera_image)
            else:
                rgb, depth, predicted_seg, camera_pose, camera_matrix = (
                    camera_image.rgbPixels,
                    camera_image.depthPixels,
                    camera_image.segmentationMaskBuffer,
                    camera_image.camera_pose,
                    camera_image.camera_matrix,
                )
                camera_image = CameraImage(
                    rgb, depth, seg_from_gt(predicted_seg), camera_pose, camera_matrix
                )

                if self.args.save:
                    save_camera_images(camera_image)

        else:
            camera_link = link_from_name(self.robot, self.robot.CAMERA_OPTICAL_FRAME)
            camera_image = self.robot.controller.get_segmented_image(self.seg_network)
            rgb, depth, seg, _, matrix = camera_image

            self.robot.update_conf()
            camera_pose = get_link_pose(self.robot, camera_link)
            camera_image = CameraImage(rgb, depth, seg, camera_pose, matrix)

            if self.args.save:
                save_camera_images(camera_image)

        return camera_image

    def update_rendered_image(self, **kwargs):
        return self.robot.cameras[0].get_image(**kwargs)

    def estimate_state(self, task, max_attempts=5):
        self.reset_belief()

        real_image = self.get_image()

        surfaces = self.estimate_surfaces(real_image, task)
        table = surfaces[0]
        objects = self.estimate_objects(real_image, table)

        self.estimates.append(
            {
                # TODO: store the mesh *.obj files
                "date": datetime.datetime.now(),
                "surfaces": surfaces,
                "objects": objects,
            }
        )

        return self.belief

    def estimate_surfaces(self, camera_image, task):
        surfaces = estimate_surfaces(
            self.belief,
            camera_image,
            min_z=self.robot.min_z,
            max_depth=self.robot.max_depth,
            client=self.client,
        )
        return surfaces

    def estimate_objects(self, camera_image, table):
        objects = self.belief.estimate_objects(
            camera_image,
            use_seg=self.args.segmentation,
            surface=table.surface,
            project_base=not self.args.disable_project,
            sc_network=self.sc_network,
            save_relabeled_img=False,
            concave=not self.args.convex,
            surfaces_movable=True,
            max_depth=self.robot.max_depth,
        )
        return objects

    def predstate_command(self, command):
        state = WorldState(client=self.client)
        saver = WorldSaver(bodies=[body.body for body in self.belief.estimated_objects])
        before_poses = {
            obj: get_pose(obj.body) for obj in self.belief.estimated_objects
        }
        belief_sim = iterate_sequence(state, command, time_step=0)
        after_poses = {obj: get_pose(obj.body) for obj in self.belief.estimated_objects}
        tform_from_rel_pose = lambda p1, p2: multiply(p1, invert(p2))
        tform_poses = {
            obj: tform_from_rel_pose(after_poses[obj], before_poses[obj])
            for obj in self.belief.estimated_objects
        }
        camera_image_pred = self.belief.robot.cameras[0].get_image()
        saver.restore()
        self.pred_state = camera_image_pred
        self.pred_tform = tform_poses

    ##################################################

    def plan(self, belief, task, serialize=False, save=True):
        start_time = time.time()
        print(SEPARATOR)

        # state = WorldState()
        objects = belief.estimated_objects
        solution = plan_pddlstream(
            belief,
            task,
            objects=objects,
            grasp_mode=self.args.grasp_mode,
            serialize=serialize,
            debug=self.args.debug,
            client=self.client,
        )
        plan, cost, _ = solution
        sequence = post_process(plan)
        # belief.reset()

        if not save:
            return sequence

        failure = sequence is None
        self.plans.append(
            {
                "date": datetime.datetime.now(),
                "task": task,
                "serialize": serialize,  # TODO: flag
                "failure": failure,
                "plan": plan,
                "length": INF if failure else len(sequence),
                "cost": cost,
                "sequence": sequence,
                # 'goal': None,
            }
        )
        self.runtimes.setdefault("planning", []).append(elapsed_time(start_time))

        return sequence

    def execute_command(self, command, save=True):
        start_time = time.time()

        data = {}
        aborted = False
        state = WorldState(client=self.client)
        executed_commands = []
        if command is None:
            status = FAILURE_STATUS
        elif len(command) == 0:
            status = SUCCESS_STATUS
        else:
            status = ONGOING_STATUS
            if not self.args.real_execute:
                state.assign()
                iterate_sequence(state, command, teleport=self.teleport)
                aborted = False
            else:
                aborted = not command.execute(self.robot.controller)

            self.executed = True
            data.update(
                {
                    "command": command,
                }
            )

        conf = self.robot.update_conf()

        if not save:
            return status, aborted

        data.update(
            {
                "date": datetime.datetime.now(),
                "status": status,
                "failure": status is FAILURE_STATUS,
                "success": status is SUCCESS_STATUS,
                "aborted": aborted,
                "conf": conf,
                "executed": executed_commands,
            }
        )
        self.executions.append(data)
        self.runtimes.setdefault("execution", []).append(elapsed_time(start_time))

        return status, aborted

    def save_data(self):
        date_name = time.time()
        filename = "run_{}.pkl".format(date_name)
        path_name = os.path.join(self.data_folder, filename)

        # TODO: KeyboardInterrupt if the event that I kill it
        data = {
            "args": self.args,
            "runtimes": self.runtimes,
            "estimates": self.estimates,
            "renders": self.renders,
            "plans": self.plans,
            "executions": self.executions,
        }
        self.data.append(data)

        write_pickle(path_name, data)

        print("Saved data to {}".format(path_name))

    def run(
        self,
        task,
        num_iterations=INF,
        always_save=True,
        terminate=not GRASP_EXPERIMENT,
        client=None,
        **kwargs
    ):
        print("=" * 30)
        start_time = time.time()
        for iteration in irange(num_iterations):  # TODO: max time?
            self.robot.reset()
            belief = self.estimate_state(task)
            if not self.args.debug:  # Intentional
                wait_if_gui(client=client)

            sequence = self.plan(belief, task)

            self.robot.reset()
            belief.reset()
            p.removeAllUserDebugItems()

            print("Execute?")
            wait_if_gui(client=client)
            status, aborted = self.execute_command(sequence)
            if always_save or self.executed:  # Only save if the robot does something
                self.save_data()
            if terminate:
                if status is FAILURE_STATUS:
                    break
                if status is SUCCESS_STATUS:
                    print(
                        "Iteration {}: Success ({:.3f} sec)!".format(
                            iteration, elapsed_time(start_time)
                        )
                    )
                    return True

            self.robot.controller.wait(duration=2.0)
        print(
            "Iteration {}: Failure ({:.3f} sec)!".format(
                iteration, elapsed_time(start_time)
            )
        )
        wait_if_gui(client=client)  # TODO: reduce PyBullet and GPU spam output
        return False

    def run_exploration(
        self,
        task,
        num_iterations=INF,
        always_save=True,
        room=None,
        real_world=None,
        client=None,
        base_planner=None,
        **kwargs
    ):
        env = Environment(client=self.client)
        # From init of simple navigation
        env.start = (0, 0, 0)
        env.goal = (
            0,
            0,
            np.pi * 2.0 - np.pi / 2.0,
        )  # TODO: Create separate class for configuration space
        env.objects = []
        env.viewed_voxels = []

        # Properties represented as a list of width, length, height, mass
        env.objects_prop = dict()

        i = np.random.randint(-1, 4, size=2)
        env.start = (
            round(env.start[0] + i[0] * GRID_RESOLUTION, 2),
            round(env.start[1] + i[1] * GRID_RESOLUTION, 2),
            round(env.start[2] + np.random.randint(16) * np.pi / 8, 3),
        )

        i = np.random.randint(-1, 5)
        env.goal = (
            env.goal[0],
            round(env.goal[1] + i * GRID_RESOLUTION, 2),
            env.goal[2],
        )

        env.goal = (2.2, 1, env.goal[2])
        env.initialized = True

        with LockRenderer(client=self.client):
            env.set_defaults(self.robot.body, client=client)
            env.objects += real_world.room.movable_obstacles
            env.camera_pose = get_link_pose(
                self.robot.body,
                link_from_name(
                    self.robot.body, "kinect2_rgb_optical_frame", client=client
                ),
                client=self.client,
            )

            env.joints = [
                joint_from_name(self.robot.body, "x", client=client),
                joint_from_name(self.robot.body, "y", client=client),
                joint_from_name(self.robot.body, "theta", client=client),
            ]

            env.robot = self.robot.body
            env.room = room
            env.static_objects = []
            env.setup_grids()
            env.centered_aabb = env.get_centered_aabb()
            env.centered_oobb = env.get_centered_oobb()

            if not env.initialized:
                env.randomize_env()
            env.display_goal(env.goal)

            env.joints = [
                joint_from_name(env.robot, "x", client=client),
                joint_from_name(env.robot, "y", client=client),
                joint_from_name(env.robot, "theta", client=client),
            ]
            set_joint_positions(env.robot, env.joints, env.start, client=client)

        planner = base_planner(env, client=client)
        plan = planner.get_plan(loadfile=None)

        p.removeAllUserDebugItems()

        print("=" * 30)
        start_time = time.time()
        for iteration in irange(num_iterations):  # TODO: max time?
            self.robot.reset()
            belief = self.estimate_state(task)
            if self.args.debug:  # Intentional
                wait_if_gui(client=client)

            sequence = self.plan(belief, task)

            self.robot.reset()
            belief.reset()
            p.removeAllUserDebugItems()

            if self.args.debug:  # Intentional
                wait_if_gui(client=client)

            status, aborted = self.execute_command(sequence)
            if always_save or self.executed:  # Only save if the robot does something
                self.save_data()

            self.robot.controller.wait(duration=2.0)
        print(
            "Iteration {}: Failure ({:.3f} sec)!".format(
                iteration, elapsed_time(start_time)
            )
        )
        wait_if_gui(client=client)  # TODO: reduce PyBullet and GPU spam output
        return False

    ##################################################
