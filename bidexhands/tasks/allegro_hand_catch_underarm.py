# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from matplotlib.pyplot import axis
#add point cloud
from PIL import Image as Im
import numpy as np
import os
import random
import torch

from bidexhands.utils.torch_jit_utils import *
from bidexhands.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

import sys
import trimesh

class AllegroHandCatchUnderarm(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.allegro_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen", "ycb/banana", "ycb/can", "ycb/mug", "ycb/brick"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "ycb/banana": "urdf/ycb/011_banana/011_banana.urdf",
            "ycb/can": "urdf/ycb/010_potted_meat_can/010_potted_meat_can.urdf",
            "ycb/mug": "urdf/ycb/025_mug/025_mug.urdf",
            "ycb/brick": "urdf/ycb/061_foam_brick/061_foam_brick.urdf"
        }

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["point_cloud", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        print("Obs type:", self.obs_type)

        self.num_point_cloud_feature_dim = 768
        self.num_obs_dict = {
            "point_cloud": 150 + self.num_point_cloud_feature_dim * 3,
            "point_cloud_for_distill": 402 + self.num_point_cloud_feature_dim * 3,
            "full_state": 150
        }
        self.num_hand_obs = 72 + 95 + 22
        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 22
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 22

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.finger_mesh_files = {
            "if1": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f1.stl",
            "if2": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f2.stl",
            "if3": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f3.stl",
            "if4": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f4.stl",
            
            "mf1": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f1.stl",
            "mf2": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f2.stl",
            "mf3": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f3.stl",
            "mf4": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f4.stl",


            "pf1": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f1.stl",
            "pf2": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f2.stl",
            "pf3": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f3.stl",
            "pf4": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f4.stl",

            "th1": "../assets/urdf/allegro_hand_model/meshes/40_10_link_t1_r.stl",
            "th2": "../assets/urdf/allegro_hand_model/meshes/40_10_link_f1.stl",
            "th3": "../assets/urdf/allegro_hand_model/meshes/40_10_link_t3.stl",
            "th4": "../assets/urdf/allegro_hand_model/meshes/40_10_link_t4.stl",
           
        }

        self.mesh_scale_factor = 0.001
        self.link_point_cloud_sample_num = 96
        self.name2index = {}
        #add point cloud
        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) 
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        print("the shape of actor_root_state_tensor: ",actor_root_state_tensor.shape)   # (32, 13)  8 * 4 * 13
        print("the shape of dof_state_tensor: ",dof_state_tensor.shape)                 # (352, 2)  8 * 44 * 2
        print("the shape of rigid_body_tensor: ",rigid_body_tensor.shape)               # (384, 13) 8 * 48 * 13
        
        # if self.obs_type == "full_state" or self.asymmetric_obs:
        #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        #     dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        #     self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_allegro_hand_dofs * 2)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.allegro_hand_default_dof_pos = torch.zeros(self.num_allegro_hand_dofs, dtype=torch.float, device=self.device)
        self.allegro_hand_default_dof_pos[:6] = torch.tensor([0, 0, -1, 3.14, 0.57, 3.14], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.allegro_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_allegro_hand_dofs]
        self.allegro_hand_dof_pos = self.allegro_hand_dof_state[..., 0]
        self.allegro_hand_dof_vel = self.allegro_hand_dof_state[..., 1]

        # self.allegro_hand_another_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_allegro_hand_dofs:self.num_allegro_hand_dofs*2]
        # self.allegro_hand_another_dof_pos = self.allegro_hand_another_dof_state[..., 0]
        # self.allegro_hand_another_dof_vel = self.allegro_hand_another_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone() 

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        self.link_points = {}
        for link_mesh_file_key in ["if1","if2","if3", "if4", "mf1","mf2","mf3", "mf4", "pf1","pf2","pf3", "pf4", "th1", "th2", "th3", "th4"]:
            link_mesh_file = self.finger_mesh_files[link_mesh_file_key]
            link_mesh = trimesh.load(link_mesh_file, multimaterial=True)
            link_mesh.apply_scale(self.mesh_scale_factor)
            points, _ = trimesh.sample.sample_surface(link_mesh,  self.link_point_cloud_sample_num)
            if "f1" in link_mesh_file_key or "f2" in link_mesh_file_key:
                def rotate_points_90_deg_y(points):
                    points = np.array(points)
                    rotation_matrix = np.array([
                            [0,  0, 1],
                            [0,  1, 0],
                            [-1, 0, 0]
                    ])
                    rotated_points = points.dot(rotation_matrix)
                    return rotated_points.tolist()
                points = rotate_points_90_deg_y(points)
            if "h1" in link_mesh_file_key or "h2" in link_mesh_file_key:
                def rotate_points_90_deg_x(points):
                    points = np.array(points)
                    rotation_matrix = np.array([
                            [1,  0, 0],
                            [0,  0, 1],
                            [0, -1, 0]
                    ])
                    rotated_points = points.dot(rotation_matrix)
                    return rotated_points.tolist()
                points = rotate_points_90_deg_x(points)
            self.link_points[link_mesh_file_key] = points

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        allegro_hand_asset_file = "urdf/xarm_description/urdf/xarm6.urdf"
        allegro_hand_another_asset_file = "urdf/xarm_description/urdf/xarm6.urdf"

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        allegro_hand_asset = self.gym.load_asset(self.sim, asset_root, allegro_hand_asset_file, asset_options)
        

        self.num_allegro_hand_bodies = self.gym.get_asset_rigid_body_count(allegro_hand_asset)
        self.num_allegro_hand_shapes = self.gym.get_asset_rigid_shape_count(allegro_hand_asset)
        self.num_allegro_hand_dofs = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_actuators = self.gym.get_asset_dof_count(allegro_hand_asset)
        self.num_allegro_hand_tendons = self.gym.get_asset_tendon_count(allegro_hand_asset)

        print("self.num_allegro_hand_bodies: ", self.num_allegro_hand_bodies)         # 23
        print("self.num_allegro_hand_shapes: ", self.num_allegro_hand_shapes)         # 28
        print("self.num_allegro_hand_dofs: ", self.num_allegro_hand_dofs)             # 22
        print("self.num_allegro_hand_actuators: ", self.num_allegro_hand_actuators)   # 22
        print("self.num_allegro_hand_tendons: ", self.num_allegro_hand_tendons)       # 0

        # tendon set up
        limit_stiffness = 3
        t_damping = 0.1
       
        tendon_props = self.gym.get_asset_tendon_properties(allegro_hand_asset)


        for i in range(self.num_allegro_hand_tendons):
            tendon_props[i].limit_stiffness = limit_stiffness
            tendon_props[i].damping = t_damping

        self.gym.set_asset_tendon_properties(allegro_hand_asset, tendon_props)
       
        
        self.actuated_dof_indices = [i for i in range(16)]

        # set allegro_hand dof properties
        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_hand_asset)
       
        self.allegro_hand_dof_lower_limits = []
        self.allegro_hand_dof_upper_limits = []
        self.allegro_hand_dof_default_pos = []
        self.allegro_hand_dof_default_vel = []
        self.allegro_hand_dof_stiffness = []
        self.allegro_hand_dof_damping = []
        self.allegro_hand_dof_effort = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_allegro_hand_dofs):
            self.allegro_hand_dof_lower_limits.append(allegro_hand_dof_props['lower'][i])
            self.allegro_hand_dof_upper_limits.append(allegro_hand_dof_props['upper'][i])
            self.allegro_hand_dof_default_pos.append(0.0)
            self.allegro_hand_dof_default_vel.append(0.0)

        for i in range(6, self.num_allegro_hand_dofs):
            allegro_hand_dof_props['stiffness'][i] = 3
            allegro_hand_dof_props['damping'][i] = 0.1
            allegro_hand_dof_props['effort'][i] = 0.5
           

        x_arm_dof_effort = to_torch([10, 10, 6, 6, 6, 4], dtype=torch.float, device=self.device)
        
        for i in range(0, 6):
            allegro_hand_dof_props['stiffness'][i] = 50
            allegro_hand_dof_props['damping'][i] = 1
            allegro_hand_dof_props['effort'][i] = x_arm_dof_effort[i]
           

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.allegro_hand_dof_lower_limits = to_torch(self.allegro_hand_dof_lower_limits, device=self.device)
        self.allegro_hand_dof_upper_limits = to_torch(self.allegro_hand_dof_upper_limits, device=self.device)
        self.allegro_hand_dof_default_pos = to_torch(self.allegro_hand_dof_default_pos, device=self.device)
        self.allegro_hand_dof_default_vel = to_torch(self.allegro_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        allegro_hand_start_pose = gymapi.Transform()
        allegro_hand_start_pose.p = gymapi.Vec3(0.55, 0, 0.6)
        allegro_hand_start_pose.r = gymapi.Quat().from_euler_zyx(3.14159, 3.14159, 0)


        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        

        if self.object_type == "pen":
            object_start_pose.p.z = allegro_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.4)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        # goal_start_pose.p = gymapi.Vec3(0, 0, 1.0)
        goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        goal_start_pose.p.z -= 0.0

        # compute aggregate size
        max_agg_bodies = self.num_allegro_hand_bodies * 2 + 2
        max_agg_shapes = self.num_allegro_hand_shapes * 2 + 2

        self.fingertips = ["mf4", "pf4", "if4", "th4"]
        self.num_fingertips = len(self.fingertips)

        self.allegro_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
       
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(allegro_hand_asset, name) for name in self.fingertips]
       

        print("fingertip_handles: ", self.fingertip_handles)
        

        # create fingertip force sensors, if needed
        # if self.obs_type == "full_state" or self.asymmetric_obs:
        #     sensor_pose = gymapi.Transform()
        #     for ft_handle in self.fingertip_handles:
        #         self.gym.create_asset_force_sensor(allegro_hand_asset, ft_handle, sensor_pose)
        #     for ft_a_handle in self.fingertip_another_handles:
        #         self.gym.create_asset_force_sensor(allegro_hand_another_asset, ft_a_handle, sensor_pose)
        
        #add point cloud
        if self.obs_type in ["point_cloud"]:
            self.cameras = []
            self.camera_tensors = []
            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
            self.camera_props.enable_tensors = True

            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.pointCloudDownsampleNum = 768
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

            self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

            if self.point_cloud_debug:
                import open3d as o3d
                from bidexhands.utils.o3dviewer import PointcloudVisualizer
                self.pointCloudVisualizer = PointcloudVisualizer()
                self.pointCloudVisualizerInitialized = False
                self.o3d_pc = o3d.geometry.PointCloud()
            else :
                self.pointCloudVisualizer = None
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(                             
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            allegro_hand_actor = self.gym.create_actor(env_ptr, allegro_hand_asset, allegro_hand_start_pose, "hand", i, -1, 0)
            if i == 0:
                allegro_hand_actor_rigid_names = self.gym.get_actor_rigid_body_names(env_ptr, allegro_hand_actor)
                for j in range(len(allegro_hand_actor_rigid_names)):
                    self.name2index[allegro_hand_actor_rigid_names[j]] = j
                    print("allegro_hand_actor_rigid_names, ", allegro_hand_actor_rigid_names)
            
           
            self.hand_start_states.append([allegro_hand_start_pose.p.x, allegro_hand_start_pose.p.y, allegro_hand_start_pose.p.z,
                                           allegro_hand_start_pose.r.x, allegro_hand_start_pose.r.y, allegro_hand_start_pose.r.z, allegro_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, allegro_hand_actor, allegro_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, allegro_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

                    

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, allegro_hand_actor)
            print("num_bodies: ", num_bodies)
            
            hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
            
            # for n in self.agent_index[0]:
            #     colorx = random.uniform(0, 1)
            #     colory = random.uniform(0, 1)
            #     colorz = random.uniform(0, 1)
            #     for m in n:
            #         for o in hand_rigid_body_index[m]:
            #             self.gym.set_rigid_body_color(env_ptr, allegro_hand_actor, o, gymapi.MESH_VISUAL,
            #                                     gymapi.Vec3(colorx, colory, colorz))
            # for n in self.agent_index[1]:                
            #     colorx = random.uniform(0, 1)
            #     colory = random.uniform(0, 1)
            #     colorz = random.uniform(0, 1)
            #     for m in n:
            #         for o in hand_rigid_body_index[m]:
            #             self.gym.set_rigid_body_color(env_ptr, allegro_hand_another_actor, o, gymapi.MESH_VISUAL,
            #                                     gymapi.Vec3(colorx, colory, colorz))
                # gym.set_rigid_body_texture(env, actor_handles[-1], n, gymapi.MESH_VISUAL,
                #                            loaded_texture_handle_list[random.randint(0, len(loaded_texture_handle_list)-1)])

            # create fingertip force-torque sensors
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, allegro_hand_actor)
                #self.gym.enable_actor_dof_force_sensors(env_ptr, allegro_hand_another_actor)
            
            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
            #add point cloud 
            if self.obs_type in ["point_cloud"]:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(-0.55, -0.3, 0.6), gymapi.Vec3(0, 0, 0.4))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)

                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_hand_actor)

        print("self.hand_indices: ", self.hand_indices)
        print("self.object_indices: ", self.object_indices)
        print("self.goal_object_indices, ", self.goal_object_indices)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        # self.fingertip_another_handles = to_torch(self.fingertip_another_handles, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        #self.another_hand_indices = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

        

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.allegro_right_hand_pos, self.right_hand_if_pos, 
            self.right_hand_mf_pos, self.right_hand_pf_pos, self.right_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        #add point cloud
        if self.obs_type in ["point_cloud"]:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        self.allegro_right_hand_pos = self.rigid_body_states[:, 6, 0:3]
        self.allegro_right_hand_rot = self.rigid_body_states[:, 6, 3:7]

        #self.allegro_left_hand_pos = self.rigid_body_states[:, 6 + 23, 0:3]
        #self.allegro_left_hand_rot = self.rigid_body_states[:, 6 + 23, 3:7]

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        #print("self.object_pose: ", self.object_pose)

        self.block_right_handle_pos = self.rigid_body_states[:, 23, 0:3]
        self.block_right_handle_rot = self.rigid_body_states[:, 23, 3:7]
        self.block_right_handle_pos = self.block_right_handle_pos + quat_apply(self.block_right_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.)
        self.block_right_handle_pos = self.block_right_handle_pos + quat_apply(self.block_right_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.0)
        self.block_right_handle_pos = self.block_right_handle_pos + quat_apply(self.block_right_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0)

        self.right_hand_pos = self.rigid_body_states[:, 6, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, 6, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        self.right_hand_if_pos = self.rigid_body_states[:, 10, 0:3]

        self.right_hand_if_rot = self.rigid_body_states[:, 10, 3:7]
        self.right_hand_if_pos = self.right_hand_if_pos + quat_apply(self.right_hand_if_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        #print(self.right_hand_if_pos.shape)
        self.right_hand_mf_pos = self.rigid_body_states[:, 14, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, 14, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_pf_pos = self.rigid_body_states[:, 18, 0:3]
        self.right_hand_pf_rot = self.rigid_body_states[:, 18, 3:7]
        self.right_hand_pf_pos = self.right_hand_pf_pos + quat_apply(self.right_hand_pf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_th_pos = self.rigid_body_states[:, 22, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, 22, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        #print(self.right_hand_th_pos.shape)
        #print("self.block_right_handle_pose: ", self.rigid_body_states[:, 23, 0:7])

        
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_linvel = self.goal_states[:, 7:10]
        self.goal_angvel = self.goal_states[:, 10:13]

        #print("self.goal_pos: ", self.goal_pos)
        self.goal_rot = self.goal_states[:, 3:7]
        
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        
        self.right_goal_pos = to_torch([0.0, 0.0, 1.0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.right_goal_rot = self.goal_states[:, 3:7]

        #add point cloud
        if self.obs_type == "full_state":
            self.compute_full_state()
        elif self.obs_type == "point_cloud":
            self.compute_point_cloud_observation()


        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        """
        Compute the observations of all environment. The observation is composed of three parts: 
        the state values of the left and right hands, and the information of objects and target. 
        The state values of the left and right hands were the same for each task, including hand 
        joint and finger positions, velocity, and force information. The detail 219-dimensional 
        observational space as shown in below:

        Index       Description
        0 - 21	    allegro  hand dof position
        22 - 43	    allegro  hand dof velocity
        44 - 95	    allegro  hand fingertip position, linear velocity, angle velocity (4 x 13)
        96 - 98     allegro  hand base position
        99 - 101    allegro  hand base rotation	
        102 - 123	allegro  hand actions
        124 - 130   object_pose
        131 - 133   object_linvel
        134 - 136   object_angvel
        137 - 143   goal_pose
        144 - 146	goal_linvel
        147 - 149   goal_angvel
        """
        # fingertip observations, state(pose and vel) + force-torque sensors
        num_ft_states = 52

        self.obs_buf[:, 0:self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos,
                                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.obs_buf[:, self.num_allegro_hand_dofs:2*self.num_allegro_hand_dofs] = self.vel_obs_scale * self.allegro_hand_dof_vel
       
        fingertip_obs_start = 44  # 168 = 157 + 11
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        
        hand_pose_start = fingertip_obs_start + 52
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + 22] = self.actions[:, :22]

        obj_obs_start = action_obs_start + 22 
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        goal_obs_start = obj_obs_start + 13  
        self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 10] = self.goal_linvel
        self.obs_buf[:, goal_obs_start + 10:goal_obs_start + 13] = self.vel_obs_scale * self.goal_angvel
        #self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        #self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        #print("the shape of self.obs_buf: ", self.obs_buf.shape)
        #sys.exit(0)

    #add point cloud
    def compute_point_cloud_observation(self, collect_demonstration=False):
        """
        Compute the observations of all environment. The observation is composed of three parts: 
        the state values of the left and right hands, and the information of objects and target. 
        The state values of the left and right hands were the same for each task, including hand 
        joint and finger positions, velocity, and force information. The detail 219-dimensional 
        observational space as shown in below:

        Index       Description
        0 - 21	    allegro  hand dof position
        22 - 43	    allegro  hand dof velocity
        44 - 95	    allegro  hand fingertip position, linear velocity, angle velocity (4 x 13)
        96 - 98     allegro  hand base position
        99 - 101    allegro  hand base rotation	
        102 - 123	allegro  hand actions
        124 - 130   object_pose
        131 - 133   object_linvel
        134 - 136   object_angvel
        137 - 143   goal_pose
        144 - 146	goal_linvel
        147 - 149   goal_angvel
        """
        # Fingertip observations, state(pose and vel) + force-torque sensors
        num_ft_states = 52
       
        self.obs_buf[:, 0:self.num_allegro_hand_dofs] = unscale(self.allegro_hand_dof_pos,
                                                            self.allegro_hand_dof_lower_limits, self.allegro_hand_dof_upper_limits)
        self.obs_buf[:, self.num_allegro_hand_dofs:2*self.num_allegro_hand_dofs] = self.vel_obs_scale * self.allegro_hand_dof_vel

        fingertip_obs_start = 44
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
       
        hand_pose_start = fingertip_obs_start + 52
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + 22] = self.actions[:, :22]

        obj_obs_start = action_obs_start + 22
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        goal_obs_start = obj_obs_start + 13 
        self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 10] = self.goal_linvel
        self.obs_buf[:, goal_obs_start + 10:goal_obs_start + 13] = self.vel_obs_scale * self.goal_angvel
        # self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        # self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        #print("the shape of self.obs_buf: ", self.obs_buf.shape)
        def rotation_matrix(w, x, y, z):
            item00 = 1.0 - 2.0 * (y * y + z * z)
            item01 = 2.0 * (x * y - w * z)
            item02 = 2.0 * (x * z + w * y)
            item10 = 2.0 * (x * y + w * z)
            item11 = 1.0 - 2.0 * (x * x + z * z)
            item12 = 2.0 * (y * z - w * x)
            item20 = 2.0 * (x * z - w * y)
            item21 = 2.0 * (y * z + w * x)
            item22 = 1.0 - 2.0 * (x * x + y * y)
            return torch.tensor([[item00, item01, item02], [item10, item11, item12], [item20, item21, item22]], dtype=torch.float, device=self.device)
        
        point_clouds_augmentation = torch.zeros((self.num_envs, self.link_point_cloud_sample_num*16, 3), dtype=torch.float, device=self.device)

        for i in range(self.num_envs):
            point_clouds_all = []
            for link_mesh_file_key in ["if1","if2","if3", "if4", "mf1","mf2","mf3", "mf4", "pf1","pf2","pf3", "pf4", "th1", "th2", "th3", "th4"]:
                points = self.link_points[link_mesh_file_key]
                link_pos = self.rigid_body_states[i, self.name2index[link_mesh_file_key], 0 : 3]
                link_rot = self.rigid_body_states[i, self.name2index[link_mesh_file_key], 3 : 7]
                #print("the shape of link_pos: ", link_rot.shape)
                quat = rotation_matrix(link_rot[3], link_rot[0], link_rot[1], link_rot[2])
                points = torch.tensor(points, dtype=torch.float, device=self.device).unsqueeze(-1)
                quat = torch.tensor(quat, dtype=torch.float, device=self.device).T.repeat(points.shape[0], 1, 1)
                quat_1 = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]], dtype=torch.float, device=self.device).repeat(points.shape[0], 1, 1)
                #print("the shape of points: ", points.shape)
                #print("the shape of quat: ", quat.shape)
                points_transformed = torch.matmul(quat, points)
                points_transformed = torch.matmul(quat_1, points_transformed)
                #print("the shape of trans: ", points_transformed.shape)
                #print("the shape of points_transformed.shape: ", points_transformed.shape)
                points_transformed = points_transformed.squeeze(-1) + link_pos.unsqueeze(0)
                #print("the shape of points_transformed: ", points_transformed.shape)
                point_clouds_all.append(points_transformed)
            point_clouds_all = torch.cat(point_clouds_all, dim=0)

            #print("the shape of point_clouds_all: ", point_clouds_all.shape)
            point_clouds_augmentation[i] = point_clouds_all

        point_clouds = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
        if self.camera_debug:
            import matplotlib.pyplot as plt
            self.camera_rgba_debug_fig = plt.figure("CAMERA_RGBD_DEBUG")
            camera_rgba_image = self.camera_visulization(is_depth_image=False)
            plt.imshow(camera_rgba_image)
            plt.pause(1e-9)
            image_save_path = "camera_debug_image.png"  
            plt.savefig(image_save_path)

        for i in range(self.num_envs):
            # Here is an example. In practice, it's better not to convert tensor from GPU to CPU
            points = depth_image_to_point_cloud_GPU(self.camera_tensors[i], self.camera_view_matrixs[i], self.camera_proj_matrixs[i], self.camera_u2, self.camera_v2, self.camera_props.width, self.camera_props.height, 10, self.device)
            
            if points.shape[0] > 0:
                selected_points = self.sample_points(points, sample_num=self.pointCloudDownsampleNum, sample_mathed='random')
            else:
                selected_points = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
            
            point_clouds[i] = selected_points

        # if self.pointCloudVisualizer != None :
        #     import open3d as o3d
        #     pp = [point_clouds[0, :, :3], point_clouds_all]
        #     pp = torch.cat(pp, dim=0)
            
        #     points = point_clouds[0, :, :3].cpu().numpy()
        #     # colors = plt.get_cmap()(point_clouds[0, :, 3].cpu().numpy())
        #     self.o3d_pc.points = o3d.utility.Vector3dVector(points)

        #     # axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        #     # o3d.visualization.draw_geometries([self.o3d_pc, axis_frame])
        #     # self.o3d_pc.colors = o3d.utility.Vector3dVector(colors[..., :3])
            
            
        #     if self.pointCloudVisualizerInitialized == False :
        #         self.pointCloudVisualizer.add_geometry(self.o3d_pc)
        #         self.pointCloudVisualizerInitialized = True
        #     else :
        #         self.pointCloudVisualizer.update(self.o3d_pc)


        self.gym.end_access_image_tensors(self.sim)
        point_clouds -= self.env_origin.view(self.num_envs, 1, 3)
        #point_clouds = point_clouds.reshape(self.num_envs, self.pointCloudDownsampleNum * 3)
        #print("the shape of point_clouds: ", point_clouds.shape)
        #sys.exit(0)
        point_clouds_start = goal_obs_start + 13
        #print(point_clouds_start)
        #print(self.obs_buf[0].shape)
        #sys.exit(0)
        self.obs_buf[:, point_clouds_start:].copy_(point_clouds.view(self.num_envs, self.pointCloudDownsampleNum * 3))
        #self.obs_buf[:, point_clouds_start:] = point_clouds
        

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 1] -= 0.6
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_allegro_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset shadow hand
        delta_max = self.allegro_hand_dof_upper_limits - self.allegro_hand_dof_default_pos
        delta_min = self.allegro_hand_dof_lower_limits - self.allegro_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_allegro_hand_dofs]

        # pos = self.allegro_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        pos = self.allegro_hand_default_dof_pos

        self.allegro_hand_dof_pos[env_ids, :] = pos
        

        self.allegro_hand_dof_vel[env_ids, :] = self.allegro_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_allegro_hand_dofs:5+self.num_allegro_hand_dofs*2]   

        
        self.prev_targets[env_ids, :self.num_allegro_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_allegro_hand_dofs] = pos


        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                                 object_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.allegro_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.allegro_hand_dof_lower_limits[self.actuated_dof_indices], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            # x-arm control
            targets = self.prev_targets[:, :6] + self.allegro_hand_dof_speed_scale * self.dt * self.actions[:, :6]
            self.cur_targets[:, :6] = tensor_clamp(targets,
                                    self.allegro_hand_dof_lower_limits[:6], self.allegro_hand_dof_upper_limits[:6])

            self.cur_targets[:, self.actuated_dof_indices + 6] = scale(self.actions[:, 6:22],
                                                                   self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
            self.cur_targets[:, self.actuated_dof_indices + 6] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 6],
                                                                          self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
            
            # targets = self.prev_targets[:, 22:28] + self.allegro_hand_dof_speed_scale * self.dt * self.actions[:, 22:28]
            # self.cur_targets[:, 22:28] = tensor_clamp(targets,
            #                         self.allegro_hand_dof_lower_limits[:6], self.allegro_hand_dof_upper_limits[:6])

            # self.cur_targets[:, self.actuated_dof_indices + 28] = scale(self.actions[:, 28:44],
            #                                                        self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])
            # self.cur_targets[:, self.actuated_dof_indices + 28] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 28],
            #                                                               self.allegro_hand_dof_lower_limits[self.actuated_dof_indices + 6], self.allegro_hand_dof_upper_limits[self.actuated_dof_indices + 6])

        self.prev_targets[:, :] = self.cur_targets[:, :]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        #print("self.actions: ", self.actions)
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.allegro_right_hand_pos[i], self.allegro_right_hand_rot[i])
                self.add_debug_lines(self.envs[i], self.allegro_left_hand_pos[i], self.allegro_left_hand_rot[i])


    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

    #add point cloud
    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]
    def sample_points(self, points, sample_num=1000, sample_mathed='furthest'):
        eff_points = points[points[:, 2]>0.04]
        if eff_points.shape[0] < sample_num :
            eff_points = points
        if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_mathed == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points
    #add point cloud
    def camera_visulization(self, is_depth_image=False):
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1)
            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                                         to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        return camera_image

#####################################################################
###=========================jit functions=========================###
#####################################################################

#add point cloud
@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    # time1 = time.time()
    depth_buffer = camera_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = camera_view_matrix_inv

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,  allegro_right_hand_pos,
    right_hand_if_pos, right_hand_mf_pos, right_hand_pf_pos, right_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # if ignore_z_rot:
    #     success_tolerance = 2.0 * success_tolerance
    hand_dist = torch.norm(object_pos - allegro_right_hand_pos, p=2, dim=-1)
    hand_finger_dist = (torch.norm(object_pos - right_hand_if_pos, p=2, dim=-1) + torch.norm(object_pos - right_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(object_pos - right_hand_pf_pos, p=2, dim=-1) 
                            + torch.norm(object_pos - right_hand_th_pos, p=2, dim=-1))
    # Orientation alignment for the cube in hand and goal cube
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    hand_dist_rew = torch.exp(-10 * hand_finger_dist)
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    up_rew = torch.zeros_like(hand_dist_rew) 
    up_rew = torch.where(hand_dist < 0.08, 3*(0.385 - goal_dist), up_rew)
    reward = 0.2 - hand_dist_rew + up_rew

    resets = torch.where(object_pos[:, 2] <= 0.3, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(hand_dist >= 0.2, torch.ones_like(resets), resets)

    # Find out which envs hit the goal and update successes count
    successes = torch.where(successes == 0, 
                    torch.where(goal_dist < 0.05, torch.ones_like(successes), successes), successes)
    
    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(object_pos[:, 2] <= 0.3, reward + fall_penalty, reward)

    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = torch.zeros_like(resets)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes).mean()

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
