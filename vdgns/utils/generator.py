import torch
import numpy as np
from tqdm import tqdm
import taichi as ti
import random
import math 
import cv2

from utils.gfs import String, Spring
import utils.visualization as viz
from taichi_utils import MPMSolver

def get_granular_simulator(material: str,
                           sim_resolution=16,
                           center=[0.5,0.5],
                           radius=[0.2,0.2],
                           friction_angle=45
                           ):
    ti.init(arch=ti.cpu)
    materials = {
        'SAND': MPMSolver.material_sand,
        'ELASTIC': MPMSolver.material_elastic,
        'SNOW': MPMSolver.material_snow,
        'WATER': MPMSolver.material_water,
        'STATIONARY': MPMSolver.material_stationary,
    }

    sim = MPMSolver(res=(sim_resolution, sim_resolution), max_num_particles=5000, padding=1, friction_angle=friction_angle)
    sim.add_ellipsoid(center=center,
                    radius=radius,
                    material=materials[material])
    
    return sim


def generate_granular_trajectory(material, 
                                 video_resolution=64,
                                 bg_color=0x000000,
                                 particle_color=0xFFFFFF,
                                 time_in_seconds=10,
                                 **kwargs):
    sim = get_granular_simulator(material, **kwargs)
    n_particles = sim.particle_info()['position'].shape[0]

    gui = ti.GUI("Taichi Elements", res=video_resolution, background_color=bg_color)

    total_steps = round(time_in_seconds / 1e-2)
    particles = np.zeros((total_steps, n_particles, 2), dtype=np.float32)
    video = np.zeros((total_steps, video_resolution, video_resolution, 4), dtype=np.float32)

    for step in range(total_steps):
        particles[step] = sim.particle_info()['position']
        gui.circles(particles[step],
                    radius=video_resolution // 32,
                    color=particle_color)
        video[step] = gui.get_image()
        gui.clear()
        sim.step(1e-2)

    gui.close()

    return particles, video
