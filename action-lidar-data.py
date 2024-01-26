import carla
import random
import pygame
import numpy as np
import open3d as o3d

import os
# run Carla
# os.startfile("C:/CARLA_0.9.14/CarlaUE4.exe")

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.load_world('Town02')


# Set up the simulator in synchronous mode 
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Set up the TM in synchronous mode
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Set a seed so behaviour can be repeated if necessary
# traffic_manager.set_random_device_seed(0)
# random.seed(0)

# We will aslo set up the spectator so we can see what we do
spectator = world.get_spectator()

#* Spawning Vehicles

# Retrieve the map's spawn points
spawn_points = world.get_map().get_spawn_points()

for s in spawn_points:
    world.debug.draw_string(s.location, 'O', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

# Select some models from the blueprint library
models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']
blueprints = []
for vehicle in world.get_blueprint_library().filter('*vehicle*'):
    if any(model in vehicle.id for model in models):
        blueprints.append(vehicle)


spawn_location = None
destination_location = random.choice(spawn_points)

# Take a random sample of the spawn points and spawn some the ego_vehicle
ego_vehicle = None
while ego_vehicle is None:
    spawn_location = random.choice(spawn_points)
    ego_vehicle = world.try_spawn_actor(random.choice(blueprints), spawn_location)

ego_vehicle.set_autopilot(False)
traffic_manager.ignore_lights_percentage(ego_vehicle, random.randint(0,50))

#* Rendering camera output and controlling vehicles with PyGame

# Render object to keep and pass the PyGame surface
class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(height,width,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

# Camera sensor callback, reshapes raw data from camera into 2D RGB and applies to PyGame surface
def pygame_callback(data, obj):

    img = np.reshape(np.copy(data.raw_data), (data.height, data.width, 4))
    img = img[:,:,:3]
    img = img[:, :, ::-1]

    # Fix display bug
    pixel_size = data.width * data.height
    width = int(np.ceil(data.width/64)*64)
    height = int(pixel_size / width)
    img = img.reshape((pixel_size, 3))
    img = img[:width*height, :].reshape((height, width, 3))

    obj.surface = pygame.surfarray.make_surface(img.swapaxes(0,1))

# Initialise the camera floating behind the vehicle
camera_init_trans = carla.Transform(carla.Location(x=-5, z=3), carla.Rotation(pitch=-20))
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Start camera with PyGame callback
camera.listen(lambda image: pygame_callback(image, renderObject))

# Get camera dimensions
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

# Fix display bug for pygame window
pixel_size = image_w * image_h
width = int(np.ceil(image_w/64)*64)
height = int(pixel_size / width)
diff = width - image_w

# Instantiate objects for rendering and vehicle control
renderObject = RenderObject(width, height)


#* Initialize PyGame Interface:

# Initialise the display
pygame.init()
gameDisplay = pygame.display.set_mode((width-diff,height), pygame.HWSURFACE | pygame.DOUBLEBUF)
# Draw black to the display
gameDisplay.fill((0,0,0))
gameDisplay.blit(renderObject.surface, (0,0))
pygame.display.flip()

# create a list of all the waypoints for the car to drive through (for visualization purposes)
# distance = 1
# waypoints = world.get_map().generate_waypoints(distance)
# for w in waypoints:
#     world.debug.draw_string(w.transform.location, 'O', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

# set up data structor for Lidar sensor data
pcd = o3d.geometry.PointCloud()

#lidar data processing function
def lidar_callback(point_cloud):
    points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # print(arr)

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])

#creating lidar
lidar_cam = None
lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('channels',str(32))
lidar_bp.set_attribute('points_per_second',str(90000))
lidar_bp.set_attribute('rotation_frequency',str(40))
lidar_bp.set_attribute('range',str(20))
lidar_location = carla.Location(0,0,2)
lidar_rotation = carla.Rotation(0,0,0)
lidar_transform = carla.Transform(lidar_location,lidar_rotation)
lidar_sen = world.spawn_actor(lidar_bp,lidar_transform,attach_to=ego_vehicle)
# lidar_sen.listen(lambda point_cloud: point_cloud.save_to_disk('tutorial/new_lidar_output/%.6d.ply' % point_cloud.frame))
lidar_sen.listen(lidar_callback)

import sys

sys.path.insert(0, "C:/CARLA_0.9.14/PythonAPI/carla")
from agents.navigation.global_route_planner import GlobalRoutePlanner
grp = GlobalRoutePlanner(world.get_map(), 2.0)

world.tick()

w1 = grp.trace_route(spawn_location.location, destination_location.location)
for (w, _) in w1:
    print(w.transform.location.distance(ego_vehicle.get_location()))
    world.debug.draw_string(w.transform.location, 'O', draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=120.0, persistent_lines=True)


#* Game loop

crashed = False

while not crashed:
    # Advance the simulation time
    world.tick()
    # Update the display
    gameDisplay.fill((0,0,0)) 
    gameDisplay.blit(renderObject.surface, (0,0))
    pygame.display.flip()
    
    # Get Control
    control = ego_vehicle.get_control()
    action = (control.throttle, control.steer, control.brake)
    
    # print action
    # print("action:", action)

    ego_location = ego_vehicle.get_location()

    # get the point that is the center of the closest lane to the vehicle
    waypoint = client.get_world().get_map().get_waypoint(ego_location, project_to_road=True)
    # world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=-1.0, persistent_lines=True)

    # get distance to the center of the lane
    dist_to_lane = ego_location.distance(waypoint.transform.location)

    # get the velocity of the vehicle (can be used to get speed and orientation)
    ego_velocity = ego_vehicle.get_velocity()

    # get lidar points
    lidar_points = np.asarray(pcd.points)

    # TODO get list of waypoints of a path to follow for the ego_vehicle with a target destination in mind

    # print state
    # print("lidar:", lidar_points)
    # print("distance:", dist_to_lane)
    # print("velocity:", ego_velocity)

    # o3d.visualization.draw_geometries([pcd]) # visualize the lidar

    # close the program
    for event in pygame.event.get():
        # If the window is closed, break the while loop
        if event.type == pygame.QUIT:
            crashed = True

# Stop camera and quit PyGame after exiting game loop
camera.stop()
camera.destroy()
lidar_sen.stop()
lidar_sen.destroy()
ego_vehicle.destroy()
pygame.quit()

# close CARLA
# os.system("taskkill /f /im CarlaUE4-Win64-Shipping.exe")

# NOTE carla will crash on close of program, so I end the CARLA task and rerun CARLA before running this file again
# Set up the simulator in synchronous mode 
# Set up the simulator in synchronous mode 
settings = world.get_settings()
settings.synchronous_mode = False # Enables synchronous mode
world.apply_settings(settings)

# Set up the TM in synchronous mode
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(False)