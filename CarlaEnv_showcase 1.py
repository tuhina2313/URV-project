import carla
import random
import pygame
import numpy as np
import open3d as o3d
import gymnasium as gym
from gymnasium import spaces
import sys
sys.path.insert(0, "C:/CARLA_0.9.14/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner

# import os
# # run Carla
# os.startfile("C:/CARLA_0.9.14/CarlaUE4.exe")

discrete_count = 2.0

discrete_actions = {}
for i in range(int(discrete_count) + 1):
    for j in range(2*int(discrete_count) + 1):
        step = (2*int(discrete_count) + 1)*i + j
        discrete_actions[step] = [round(i / discrete_count,2), round(j / discrete_count - 1.0,2)]

# for (i, keyval) in enumerate(discrete_actions.items()):
#     print(keyval)

# Carla Gym Environment for model learning
class CarlaEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, spawn, dest):
        super().__init__()
        self.action_space = spaces.Discrete(len(discrete_actions))

        # throttle, steer, speed, angle_of_car, disance_to_next_waypoint
        # angle_to_next_waypoint, angle_to_follow_after_waypoint, distance_to_destination
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, -2*np.pi, 0.0, -2*np.pi, -2*np.pi, 0.0]),
            high=np.array([1.0, 1.0, 120.0, 2*np.pi, 1000.0, 2*np.pi, 2*np.pi, 1000.0])
        )

        # get location of start and destination
        self.spawn_location = spawn
        self.destination_location = dest

        # spawn the ego_vehicle
        self.ego_vehicle = world.try_spawn_actor(random.choice(blueprints), self.spawn_location)
        self.ego_vehicle.set_autopilot(False)

        # Initialise the camera floating behind the vehicle
        camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
        
        # Start camera
        self.camera.listen(camera_callback)

        # calculate the route for the vehicle
        self.route_waypoints = []
        self.populate_route_waypoints()

        # parameters for determining if a waypoint was passed
        self.waypoint_distance = 1

        # paramaters for terminating a run
        self.max_steps = 2000
        self.cur_step = 0
        self.max_distance_from_route = 2

        # setup next waypoint to along route
        self.next_waypoint = self.route_waypoints[0]

        # set up target speed of the vehicle
        self.target_speed = 30.0

        # total_reward, speed_reward, distance_from_waypoints, reached_waypoints_reward, goal_reward, distance_from_start_reward, termination_reward
        self.reward_stats = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


        # initial adjustments for the next waypoint
        self.check_waypoint_information_with_reward() 


    def populate_route_waypoints(self):
        # calculate the route for the vehicle
        grp = GlobalRoutePlanner(world.get_map(), 2.0)
        world.tick()
        w1 = grp.trace_route(self.spawn_location.location, self.destination_location.location)


        # get information about the waypoints of the route and redefine it
        self.route_waypoints = []

        for i in range(len(w1)):
            w_location = w1[i][0].transform.location
            if(i < len(w1) - 1):
                w_location_next = w1[i+1][0].transform.location
            else:
                w_location_next = self.destination_location.location
            d = w_location.distance(w_location_next)
            if(d <= 0):
                continue
            if (w_location_next.y >= w_location.y):
                angle = np.arccos((w_location_next.x - w_location.x)/d)
            else:
                angle = -1*np.arccos((w_location_next.x - w_location.x)/d)
            self.route_waypoints.append({"location": w_location, "angle_to_next": angle, "passed": False})
        self.route_waypoints.append({
            "location": self.destination_location.location, 
            "angle_to_next": self.route_waypoints[len(self.route_waypoints)-1]["angle_to_next"], # get previouse angle to avoid issues
            "passed": False
        })
    
    def get_observation_val(self):
        # throttle, steer, speed, angle_of_car, disance_to_next_waypoint
        # angle_to_next_waypoint, angle_to_follow_after_waypoint, distance_to_destination
        control = self.ego_vehicle.get_control()
        ego_velocity = self.ego_vehicle.get_velocity()
        ego_location = self.ego_vehicle.get_location()
        throttle = control.throttle
        steer = control.steer
        speed = np.sqrt(ego_velocity.x * ego_velocity.x + ego_velocity.y * ego_velocity.y)
        angle_of_car = self.ego_vehicle.get_transform().rotation.yaw*np.pi/180.0
        disance_to_next_waypoint = self.next_waypoint["location"].distance(ego_location)
        if(self.next_waypoint["location"].y >= ego_location.y):
            angle_to_next_waypoint = np.arccos((self.next_waypoint["location"].x - ego_location.x)/disance_to_next_waypoint)
        else:
            angle_to_next_waypoint = -1*np.arccos((self.next_waypoint["location"].x - ego_location.x)/disance_to_next_waypoint)
        angle_to_follow_after_waypoint = self.next_waypoint["angle_to_next"]
        distance_to_destination = self.destination_location.location.distance(ego_location)
        return np.array([throttle, steer, speed, angle_of_car, disance_to_next_waypoint, angle_to_next_waypoint, angle_to_follow_after_waypoint, distance_to_destination])
    
    def check_if_waypoint_is_destination(self, waypoint):
        return (self.destination_location.location.distance(waypoint["location"]) <= 0)

    def check_waypoint_information_with_reward(self):
        min_dist = self.max_distance_from_route + 0.1
        reached_destination = False
        reward = 0
        for i in range(len(self.route_waypoints)):
            dist = self.route_waypoints[i]["location"].distance(self.ego_vehicle.get_location())
            min_dist = min(min_dist, dist)
            if(not self.route_waypoints[i]["passed"] and dist <= self.waypoint_distance):
                self.route_waypoints[i]["passed"] = True
                val = 1.5*self.route_waypoints[i]["location"].distance(self.destination_location.location)

                reward += val
                self.reward_stats[3] += val
                if(self.check_if_waypoint_is_destination(self.route_waypoints[i])):
                    reached_destination = True
                    reward += 2000.0
                    self.reward_stats[4] += 2000.0
                    break
                else:
                    self.next_waypoint = self.route_waypoints[i+1]
        ego_velocity = self.ego_vehicle.get_velocity()
        speed = np.sqrt(ego_velocity.x * ego_velocity.x + ego_velocity.y * ego_velocity.y)
        speed_factor = 0.0
        if(speed < self.target_speed):
            speed_factor = 1.0 - np.abs(speed - self.target_speed)/self.target_speed
        else:
            speed_factor = 5.0*(1.0 - np.abs(speed - self.target_speed)/self.target_speed)

        reward -= speed_factor*10.0
        self.reward_stats[1] -= speed_factor*10.0

        min_dist_factor = np.abs(min_dist - self.max_distance_from_route)/self.max_distance_from_route
        reward -= min_dist_factor*10.0
        self.reward_stats[2] -= min_dist_factor*10.0
        
        within_distance = min_dist <= self.max_distance_from_route
                    
        return reached_destination, reward, within_distance

    def get_waypoint_closest_to_other(self, waypoint):
        min_dist = np.inf
        closest_waypoint = self.route_waypoints[0]
        for i in range(len(self.route_waypoints)):
            dist = self.route_waypoints[i]["location"].distance(waypoint.transform.location)
            if(dist < min_dist):
                min_dist = dist
                closest_waypoint = self.route_waypoints[i]
        return closest_waypoint

    def step(self, action):

        # get control parameters for ego_vehicle from action
        arr = discrete_actions[action.item()]
        throttle = arr[0]
        steering = arr[1]

        # apply control to ego_vehicle
        control = carla.VehicleControl()
        control.throttle = throttle
        control.steer = steering
        self.ego_vehicle.apply_control(control)

        # tick the world and iterate timesteps
        world.tick()
        self.cur_step += 1

        # get infromation about vehicle with waypoints
        reached_destination, reward, within_distance = self.check_waypoint_information_with_reward()
        terminated = reached_destination or (not within_distance)

        # check to see if we reach the max steps
        truncated = False
        if(self.cur_step >= self.max_steps):
            truncated = True
        
        if((not within_distance) or truncated):

            # give negative reward based on if car is facing correct direction
            ego_location = self.ego_vehicle.get_location()
            waypoint = client.get_world().get_map().get_waypoint(ego_location, project_to_road=True)
            closest = self.get_waypoint_closest_to_other(waypoint)
            angle_to_next = closest["angle_to_next"]
            angle_of_car = self.ego_vehicle.get_transform().rotation.yaw*np.pi/180.0
            diff_in_angle = np.abs(angle_to_next - angle_of_car)
            diff_factor = np.abs(np.sin(diff_in_angle))
            reward -= 1000.0*diff_factor + 200.0
            self.reward_stats[6] -= 1000.0*diff_factor + 200.0

            # give a reward for how far the car went
            val = 10.0*self.spawn_location.location.distance(self.ego_vehicle.get_location())
            reward += val
            self.reward_stats[5] += val


        self.reward_stats[0] += reward

        if(terminated or truncated):
            print(f"total: {self.reward_stats[0]}, speed: {self.reward_stats[1]}, dist_w: {self.reward_stats[2]}, reach_w: {self.reward_stats[3]}, goal: {self.reward_stats[4]}, start: {self.reward_stats[5]}, term: {self.reward_stats[6]}")

        # get observation data from ego_vehicle
        observation = self.get_observation_val()

        # return more info (not used in our case)
        info = {}

        self.render()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # destory current vehicle and camera
        self.camera.stop()
        self.camera.destroy()
        self.ego_vehicle.destroy()

        # spawn the ego_vehicle
        self.ego_vehicle = world.try_spawn_actor(random.choice(blueprints), self.spawn_location)
        self.ego_vehicle.set_autopilot(False)

        # Initialise the camera floating behind the vehicle
        camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
        
        # Start camera
        self.camera.listen(camera_callback)

        # reset current steps
        self.cur_step = 0

        # calculate the route for the vehicle
        self.route_waypoints = []
        self.populate_route_waypoints()

        # setup next waypoint to along route
        self.next_waypoint = self.route_waypoints[0]

        # total_reward, speed_reward, distance_from_waypoints, reached_waypoints_reward, goal_reward, distance_from_start_reward, termination_reward
        self.reward_stats = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # initial adjustments for the next waypoint
        self.check_waypoint_information_with_reward() 

        # get observation data from new ego_vehicle
        observation = self.get_observation_val()

        return observation, {} # return observation, info

    def render(self):
        spectator.set_transform(self.camera.get_transform())

        world.debug.draw_string(self.spawn_location.location, 'O', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=-1, persistent_lines=True)
        world.debug.draw_string(self.destination_location.location, 'O', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=-1, persistent_lines=True)
        for (w) in self.route_waypoints:
            if w["passed"]:
                c = carla.Color(r=0, g=255, b=0)
            elif (w == self.next_waypoint):
                c = carla.Color(r=255, g=0, b=0)
            else:
                c = carla.Color(r=0, g=0, b=255)
            world.debug.draw_string(w["location"], 'O', draw_shadow=False, color=c, life_time=-1, persistent_lines=True)

    def close(self):
        self.camera.stop()
        self.camera.destroy()
        self.ego_vehicle.destroy()



def camera_callback(image):
    pass

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.load_world('Town07')

# Set up the simulator in synchronous mode 
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)


# We will aslo set up the spectator so we can see what we do
spectator = world.get_spectator()

# Retrieve the map's spawn points
spawn_points = world.get_map().get_spawn_points()
for (i, s) in enumerate(spawn_points):
    # print(i)
    world.debug.draw_string(s.location, str(i), draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)


# Select some models from the blueprint library
models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']
blueprints = []
for vehicle in world.get_blueprint_library().filter('*vehicle*'):
    if any(model in vehicle.id for model in models):
        blueprints.append(vehicle)



spawn_location = spawn_points[36]
destination_location = spawn_points[54]

from stable_baselines3 import PPO, A2C, DQN

env = CarlaEnv(spawn_location,  destination_location)

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=500000)

# model.save("carla_waypoint_reward")

# del model

model = PPO.load("carla_waypoint_reward")
model.set_env(env)

obs, _ = env.reset()
env.render()

done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
    total_reward += reward
env.close()
settings = world.get_settings()
settings.synchronous_mode = False # Enables synchronous mode
world.apply_settings(settings)