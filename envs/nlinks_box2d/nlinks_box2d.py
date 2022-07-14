""" 
A Box2D environment of a planar manipulator with N links 
Box2D only accepts speed control of the joints not force/torque. 
""" 
 
import time 
import Box2D 
import gym 
from Box2D.b2 import (circleShape, polygonShape, fixtureDef, revoluteJointDef, weldJointDef) 
from gym import spaces 
from gym.utils import seeding, EzPickle 
from .helper_fcts import * 
from gym.envs.classic_control import rendering 
import numpy as np 
import torchvision.transforms.functional as F 
import torch
from stable_baselines3.common import logger

gym.logger.set_level(40)

# Slow rendering with output 
DEBUG = False 
FPS = 50  # Frames per Second 
 
 
class NLinksBox2D(gym.Env, EzPickle): 
    metadata = { 
        'render.modes': ['human', 'rgb_array'], 
        'video.frames_per_second': FPS 
    } 
 
    # Punishing energy (integral of applied forces); huge impact on performance 
    POWER = False 
 
    LINK_MASS = 1.0  # kg 
    LINK_HEIGHT = 0.5  # m 
    LINK_WIDTH = 0.10  # m 
 
    MAX_TIME_STEP = FPS * 5  # Maximum length of an episode 
 
    # Min distance for end effector target 
    MIN_DISTANCE = 0.2  # m 
 
    MAX_JOINT_VEL = np.pi 
 
    GRAVITY = -9.81  # m/s^2 
 
    FIX_CIRCLE = fixtureDef( 
        shape=circleShape(radius=LINK_WIDTH), 
        density=1e-3, 
        friction=0.0, 
        restitution=0.0, 
        categoryBits=0x0020, 
        maskBits=0x001) 
 
    FIX_POLY = fixtureDef( 
        shape=polygonShape(box=(LINK_WIDTH / 2, LINK_HEIGHT / 2)), 
        density=LINK_MASS / (LINK_WIDTH * LINK_HEIGHT), 
        friction=0.0, 
        restitution=0.0, 
        categoryBits=0x0020, 
        maskBits=0x001) 
 
    time_step = n_links = np_random = world = viewer = None 
    draw_list = [] 
 
    # The Box2D objects below need to be destroyed when resetting world to release allocated memory 
    anchor = None  # static body to hold the manipulator base fixed 
    target = None  # static body to represent the target position 
    end_effector = None  # dynamic body of end effector 
    links = []  # dynamic body of links 
    joint_bodies = []  # dynamic body of joints 
    joint_fixes = []  # weld joints to connect previous link to the next joint_body 
    joints = []  # revolute joints to connect previous joint_body to the next link 
 
    def __init__(self,  
        n_links,
        obs_mode='numeric', 
        use_velocity=True, 
        sparse_rewards=False, 
        normalize_obs=True,
        seed=None,
        is_eval_env: bool = False,
        viewport_height=100, 
        viewport_width=100, 
        viewport_scale=40, 
        use_frame_stacking=False): 
 
        self.obs_mode = obs_mode 
        self.use_velocity = use_velocity 
        self.sparse_rewards = sparse_rewards 
        self.normalize_obs = normalize_obs 
        self.viewport_height = viewport_height 
        self.viewport_width = viewport_width 
        self.viewport_scale = viewport_scale 
        self.use_frame_stacking = use_frame_stacking 
 
        self.state = None 
 
        self.COLOR_ANCHOR = None 
        self.COLOR_JOINTS = None 
        self.COLOR_BORDER = None 
        self.COLOR_LINKS = None 
        self.COLOR_EE = None 
        self.COLOR_TARGET = None 
        self.COLOR_BACKGROUND = None 
 
        self.MAX_JOINT_TORQUE = None  # 80 
 
        self.ANCHOR_X = None 
        self.ANCHOR_Y = None 
 
        self.n_links = n_links 
        self.numeric_features_dim = 2 + self.n_links * (2 if self.use_velocity else 1) 
        self.is_eval_env = is_eval_env
 
        EzPickle.__init__(self) 
        self.seed(seed) 
 
        self.world = Box2D.b2World(gravity=(0, self.GRAVITY)) 
 
        self.max_ee_dist = self.n_links * self.LINK_HEIGHT 
        self.max_distance_to_target = 2 * self.max_ee_dist 
 
        # dx,dy observation space is twice as big to include max possible distance of target and end effector 
        high = [self.max_ee_dist, self.max_ee_dist] 
         
        for _ in range(self.n_links): 
            high.append(np.pi) # angle 
         
            if self.use_velocity: 
                high.append(self.MAX_JOINT_VEL) # velocity 
         
        high = np.array(high) 
 
        if self.normalize_obs: 
            high /= high 
         
        if self.obs_mode == 'numeric': 
            self.observation_space = spaces.Box(-high, high, dtype=np.float32) 
        elif self.obs_mode == 'images':
            self.observation_space = spaces.Box( 
                low=0, high=1 if self.normalize_obs else 255, 
                shape=(self.viewport_height, self.viewport_width, 3),  
                dtype=np.float32 if self.normalize_obs else np.uint8) 
        elif self.obs_mode == 'combined': 
            # append a 4th channel to the image that stores the numeric observations 
            # bit of a hack, yes 
            self.observation_space = spaces.Box( 
                low=-1, high=1 if self.normalize_obs else 255, 
                shape=(self.viewport_height, self.viewport_width, 4))
 
        self.frame_stack = np.zeros((self.viewport_height, self.viewport_width, 3)) 
 
        # action space is between -1 and 1 as recommended by stable baselines, it should be multiplied by 
        # MAX_JOINT_VEL to get the actual action (joint velocity) when controlling the manipulator 
        self.action_space = spaces.Box(-np.ones(self.n_links), np.ones(self.n_links), dtype=np.float32) 
 
        self.reset() 
 
    def _destroy(self): 
        for i in range(len(self.joints)): 
            self.world.DestroyJoint(self.joints[i]) 
        self.joints = [] 
 
        for i in range(len(self.joint_fixes)): 
            self.world.DestroyJoint(self.joint_fixes[i]) 
        self.joint_fixes = [] 
 
        for i in range(len(self.joint_bodies)): 
            self.world.DestroyBody(self.joint_bodies[i]) 
        self.joint_bodies = [] 
 
        for i in range(len(self.links)): 
            self.world.DestroyBody(self.links[i]) 
        self.links = [] 
 
        if self.end_effector: 
            self.world.DestroyBody(self.end_effector) 
        self.end_effector = None 
 
        if self.target: 
            self.world.DestroyBody(self.target) 
        self.target = None 
 
        if self.anchor: 
            self.world.DestroyBody(self.anchor) 
        self.anchor = None 
 
    def _create_anchor(self): 
        self.anchor = self.world.CreateStaticBody(position=(self.ANCHOR_X, self.ANCHOR_Y), fixtures=self.FIX_CIRCLE) 
        self.anchor.color1 = self.COLOR_ANCHOR 
        self.anchor.color2 = self.COLOR_BORDER 
        # print(f"anchor mass:         {self.anchor.mass}") 
        # print(f"anchor inertia:      {self.anchor.inertia}") 
        # print(f"anchor local center: {self.anchor.localCenter}") 
        # print(f"anchor world center: {self.anchor.worldCenter}") 
 
    def _create_first_arm(self): 
        self.joint_bodies.append( 
            self.world.CreateDynamicBody(position=self.anchor.position, fixtures=self.FIX_CIRCLE)) 
        self.joint_bodies[0].color1 = self.COLOR_JOINTS 
        self.joint_bodies[0].color2 = self.COLOR_BORDER 
 
        rjd = weldJointDef(bodyA=self.anchor, bodyB=self.joint_bodies[0], 
                           localAnchorA=(0.0, 0.0), localAnchorB=(0.0, 0.0)) 
        self.joint_fixes.append(self.world.CreateJoint(rjd)) 
 
        self.links.append( 
            self.world.CreateDynamicBody(position=self.joint_bodies[0].position - (0.0, self.LINK_HEIGHT / 2), 
                                         angle=0.0, fixtures=self.FIX_POLY)) 
        self.links[0].color1 = self.COLOR_LINKS 
        self.links[0].color2 = self.COLOR_BORDER 
        # print(f"link1 mass:         {self.links[0].mass}") 
        # print(f"link1 inertia:      {self.links[0].inertia}") 
        # print(f"link1 local center: {self.links[0].localCenter}") 
        # print(f"link1 world center: {self.links[0].worldCenter}") 
 
        rjd = revoluteJointDef(bodyA=self.anchor, bodyB=self.links[0], localAnchorA=(0.0, 0.0), 
                               localAnchorB=(0.0, self.LINK_HEIGHT / 2), 
                               enableMotor=True, maxMotorTorque=self.MAX_JOINT_TORQUE, motorSpeed=0.0) 
        self.joints.append(self.world.CreateJoint(rjd)) 
 
    def _create_next_arm(self): 
        self.joint_bodies.append( 
            self.world.CreateDynamicBody( 
                position=self.links[-1].position - (0.0, self.LINK_HEIGHT / 2), fixtures=self.FIX_CIRCLE)) 
        self.joint_bodies[-1].color1 = self.COLOR_JOINTS 
        self.joint_bodies[-1].color2 = self.COLOR_BORDER 
 
        rjd = weldJointDef(bodyA=self.links[-1], bodyB=self.joint_bodies[-1], 
                           localAnchorA=(0.0, -self.LINK_HEIGHT / 2), localAnchorB=(0.0, 0.0)) 
        self.joint_fixes.append(self.world.CreateJoint(rjd)) 
 
        self.links.append( 
            self.world.CreateDynamicBody(position=self.joint_bodies[-1].position - (0.0, self.LINK_HEIGHT / 2), 
                                         angle=0.0, fixtures=self.FIX_POLY)) 
        self.links[-1].color1 = self.COLOR_LINKS 
        self.links[-1].color2 = self.COLOR_BORDER 
        # print(f"link1 mass:         {self.links[-1].massData.mass}") 
        # print(f"link1 inertia:      {self.links[-1].massData.I}") 
        # print(f"link1 local center: {self.links[-1].localCenter}") 
        # print(f"link1 world center: {self.links[-1].worldCenter}") 
 
        rjd = revoluteJointDef(bodyA=self.links[-2], bodyB=self.links[-1], localAnchorA=(0.0, -self.LINK_HEIGHT / 2), 
                               localAnchorB=(0.0, self.LINK_HEIGHT / 2), 
                               enableMotor=True, maxMotorTorque=self.MAX_JOINT_TORQUE, motorSpeed=0.0) 
        self.joints.append(self.world.CreateJoint(rjd)) 
 
    def _create_end_effector(self): 
        self.end_effector = self.world.CreateDynamicBody(position=self.links[-1].position - (0.0, self.LINK_HEIGHT / 2), 
                                                         fixtures=self.FIX_CIRCLE) 
        self.end_effector.color1 = self.COLOR_EE 
        self.end_effector.color2 = self.COLOR_BORDER 
 
        rjd = weldJointDef(bodyA=self.links[-1], bodyB=self.end_effector, 
                           localAnchorA=(0.0, -self.LINK_HEIGHT / 2), localAnchorB=(0.0, 0.0)) 
        self.joint_fixes.append(self.world.CreateJoint(rjd)) 
 
    def _create_target(self): 
        self.target = self.world.CreateStaticBody(position=(self._random_point()), fixtures=self.FIX_CIRCLE) 
        self.target.color1 = self.COLOR_TARGET 
        self.target.color2 = self.COLOR_BORDER 
 
    def _random_point(self): 
        d = self.np_random.uniform(low=-self.max_ee_dist, high=self.max_ee_dist) 
        angle = self.np_random.uniform(low=-np.pi, high=np.pi) 
        x = d * np.cos(angle) 
        y = d * np.sin(angle) 
 
        return x, y 
 
    def _calc_power(self, action_arr): 
        power = 0 
        for action in action_arr: 
            power += abs(action) 
 
        return power 
 
    def _get_distance(self): 
        pos1 = np.array([self.end_effector.position[0], self.end_effector.position[1]]) 
        pos2 = np.array([self.target.position[0], self.target.position[1]]) 
        distance = np.linalg.norm(pos1 - pos2) 
        return distance 
 
    def _get_terminal(self): 
        return self._get_distance() < self.MIN_DISTANCE 
 
    def reset(self): 
        self.time_step = 0 
 
        self._destroy() 
        self.set_episode_parameters() 
        self._create_anchor() 
        self._create_first_arm() 
         
        for _ in range(self.n_links - 1): 
            self._create_next_arm() 
         
        self._create_end_effector() 
        self._create_target() 
 
        self.links[self.n_links - 1].ground_contact = False 
        self.draw_list = [self.anchor] + self.links + self.joint_bodies + [self.end_effector] + [self.target] 
        self.state = self._get_state()
        self.frame_stack *= 0
 
        return self._get_obs() 
 
    def seed(self, seed=None): 
        self.np_random, seed = seeding.np_random(seed) 
        return [seed] 
 
    def _get_state(self): 
        tx = self.target.position[0] 
        ty = self.target.position[1] 
 
        # eex = self.end_effector.position[0] 
        # eey = self.end_effector.position[1] 
        # dx, dy = tx - eex, ty - eey 
 
        if self.normalize_obs: 
            tx /= self.max_ee_dist 
            ty /= self.max_ee_dist 
 
        state = [tx, ty] 
 
        for idx in range(self.n_links): 
            joint_pos = self.joints[idx].angle % (np.pi * 2) - np.pi 
            joint_vel = self.joints[idx].speed 
 
            if self.normalize_obs: 
                joint_pos /= np.pi 
                joint_vel /= self.MAX_JOINT_VEL 
 
            state.append(joint_pos) 
 
            if self.use_velocity: 
                state.append(joint_vel) 
 
        return state 
 
    def _get_obs(self): 
        if self.obs_mode == 'numeric': 
            return self._get_state() 
 
        image = self.render(mode='rgb_array') 
         
        if self.obs_mode == 'images': 
            return image 
        else: 
            # combined numeric and image observations 
            obs = np.zeros((self.viewport_height, self.viewport_width, 4)) 
            obs[:,:,0:3] = image
             
            numeric_obs = self._get_state() 
            numeric_obs_dim = len(numeric_obs) 
            obs[0:numeric_obs_dim, 0, 3] = numeric_obs
            return obs
 
    def step(self, action): 
        self.time_step += 1
        speeds = action
 
        for idx in range(len(self.joints)):
            self.joints[idx].motorSpeed = float(self.MAX_JOINT_VEL * np.clip(speeds[idx], -1, 1)) 
 
        self.world.Step(1.0 / FPS, 6 * 300, 2 * 300) 
        self.world.ClearForces() 
 
        solved = self._get_terminal()
        reached_max_ts = self.time_step > self.MAX_TIME_STEP
        reward = 300 if solved else -self._get_distance()
        done = solved or reached_max_ts
 
        if done:
            domain = 'eval' if self.is_eval_env else 'rollout'
            logger.record_mean(key=domain + '/solved_fraction', value=solved)

        obs = self._get_obs()
        return obs, reward, done, {} 
 
    def render(self, mode='human'): 
        if self.viewer is None:
            self.viewer = rendering.Viewer(
                int(self.viewport_width), 
                int(self.viewport_height)) 
 
        dim = self.viewport_width / self.viewport_scale / 2 
        self.viewer.set_bounds(-dim, dim, -dim, dim) 
 
        self.viewer.draw_polygon([ 
            (-self.viewport_height, -self.viewport_height), 
            (self.viewport_height, -self.viewport_height), 
            (self.viewport_height, self.viewport_height), 
            (-self.viewport_height, self.viewport_height), 
        ], color=self.COLOR_BACKGROUND)
 
        for obj in self.draw_list: 
            for f in obj.fixtures:
                trans = f.body.transform 
                if type(f.shape) is circleShape: 
                    t = rendering.Transform(translation=trans * f.shape.pos) 
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t) 
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2).add_attr(t) 
                else: 
                    path = [trans * v for v in f.shape.vertices] 
                    self.viewer.draw_polygon(path, color=obj.color1) 
                    path.append(path[0]) 
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2) 
 
        img = self.viewer.render(return_rgb_array=mode == 'rgb_array') 
 
        if not isinstance(img, bool) and self.normalize_obs:
            img = img / 255.0 
                 
        return img 
 
    def close(self): 
        if self.viewer is not None: 
            self.viewer.close() 
            self.viewer = None 
 
    def set_episode_parameters(self): 
        self.COLOR_ANCHOR = (0., 0., 0.) 
        self.COLOR_JOINTS = (0.6, 0.6, .8) 
        self.COLOR_BORDER = (0., 0., 0.) 
        self.COLOR_LINKS = (.6, .6, 1.) 
        self.COLOR_EE = (.6, 1., .6) 
        self.COLOR_TARGET = (1., 0.6, 0.6) 
        self.COLOR_BACKGROUND = (0.9, 0.9, 1.0) 
        # self.update_visuals() 
 
        self.MAX_JOINT_TORQUE = 10000  # 80 
 
        self.ANCHOR_X = 0.0 
        self.ANCHOR_Y = 0.0 
 
