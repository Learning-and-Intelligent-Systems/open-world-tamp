from __future__ import print_function

import datetime
import time
import warnings

warnings.filterwarnings("ignore")

import os
import pickle

import numpy as np

import owt.pb_utils as pbu
from owt.estimation.belief import GRASP_EXPERIMENT, Belief
from owt.estimation.dnn import init_seg
from owt.estimation.geometry import cloud_from_depth
from owt.estimation.observation import save_camera_images
from owt.estimation.tables import estimate_surfaces
from owt.planning.planner import (iterate_sequence, plan_pddlstream,
                                  post_process)
from owt.planning.primitives import WorldState
from owt.simulation.entities import Robot, get_label_counts
from owt.simulation.tasks import Task

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
    seg_network, camera_image, fuse=False, use_depth=False, num_segs=1, **kwargs
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
    return pbu.CameraImage(rgb, depth, predicted_seg, *camera_image[3:])


class Policy(object):
    def __init__(
        self, args, robot: Robot, known=[], teleport=False, client=None, **kwargs
    ):
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

        self.belief = Belief(
            self.robot,
            surface_beliefs=[],
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
                camera_image = pbu.CameraImage(
                    rgb, depth, seg_from_gt(predicted_seg), camera_pose, camera_matrix
                )

                if self.args.save:
                    save_camera_images(camera_image)

        else:
            camera_link = pbu.link_from_name(
                self.robot, self.robot.CAMERA_OPTICAL_FRAME
            )
            camera_image = self.robot.controller.get_segmented_image(self.seg_network)
            rgb, depth, seg, _, matrix = camera_image

            self.robot.update_conf()
            camera_pose = pbu.get_link_pose(self.robot, camera_link)
            camera_image = pbu.CameraImage(rgb, depth, seg, camera_pose, matrix)

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
            save_relabeled_img=False,
            concave=not self.args.convex,
            surfaces_movable=True,
            max_depth=self.robot.max_depth,
        )
        return objects

    def predstate_command(self, command):
        state = WorldState(client=self.client)
        saver = pbu.WorldSaver(
            bodies=[body.body for body in self.belief.estimated_objects]
        )
        before_poses = {
            obj: pbu.get_pose(obj.body) for obj in self.belief.estimated_objects
        }
        belief_sim = iterate_sequence(state, command, time_step=0)
        after_poses = {
            obj: pbu.get_pose(obj.body) for obj in self.belief.estimated_objects
        }
        tform_from_rel_pose = lambda p1, p2: pbu.multiply(p1, pbu.invert(p2))
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
                "length": np.inf if failure else len(sequence),
                "cost": cost,
                "sequence": sequence,
                # 'goal': None,
            }
        )
        self.runtimes.setdefault("planning", []).append(pbu.elapsed_time(start_time))

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
        self.runtimes.setdefault("execution", []).append(pbu.elapsed_time(start_time))

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

        with open(path_name, "wb") as f:
            pickle.dump(data, f)

        print("Saved data to {}".format(path_name))

    def run(
        self,
        task: Task,
        num_iterations=np.inf,
        always_save=True,
        terminate=not GRASP_EXPERIMENT,
        client=None,
        **kwargs
    ):
        print("=" * 30)
        start_time = time.time()
        for iteration in range(num_iterations):  # TODO: max time?
            self.robot.reset()
            belief = self.estimate_state(task)
            if not self.args.debug:  # Intentional
                pbu.wait_if_gui(client=client)

            sequence = self.plan(belief, task)

            self.robot.reset()
            belief.reset()
            pbu.remove_all_debug(client=self.robot.client)

            print("Execute?")
            pbu.wait_if_gui(client=client)
            status, aborted = self.execute_command(sequence)
            if always_save or self.executed:  # Only save if the robot does something
                self.save_data()
            if terminate:
                if status is FAILURE_STATUS:
                    break
                if status is SUCCESS_STATUS:
                    print(
                        "Iteration {}: Success ({:.3f} sec)!".format(
                            iteration, pbu.elapsed_time(start_time)
                        )
                    )
                    return True

            self.robot.wait(duration=2.0)
        print(
            "Iteration {}: Failure ({:.3f} sec)!".format(
                iteration, pbu.elapsed_time(start_time)
            )
        )
        pbu.wait_if_gui(client=client)  # TODO: reduce PyBullet and GPU spam output
        return False
