import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm


def simulation_to_plot(string, time_in_seconds=2):
    fig, ax = plt.subplots(1, 5, sharey=True, figsize=(10, 2))
    idx = 0

    total_steps = int(time_in_seconds / string.get_timestep())
    draw_every = (total_steps) // 5
    for i in range(total_steps):
        if i % draw_every == 0:
            particles = np.float32(string.get_particles())
            ax[idx].plot(particles[:, 0], particles[:, 1])
            ax[idx].set_xlim(-1, 1)
            ax[idx].set_ylim(-1, 1)
            idx += 1
        string.step()


def particle_sample_to_plot(rollout):
    fig, ax = plt.subplots(1, 5, sharey=True, figsize=(10, 2))
    idx = 0

    total_steps = rollout.shape[0]
    draw_every = (total_steps) // 5
    for i in range(total_steps):
        if i % draw_every == 0:
            particles = rollout[i]
            ax[idx].plot(particles[:, 0], particles[:, 1])
            ax[idx].set_xlim(particles[0][0], particles[-1][0])
            ax[idx].set_ylim(particles[0][0], particles[-1][0])
            idx += 1


def simulation_to_video(string, output='out.mp4', time_in_seconds=3, fps=25, resolution=64, verbose=False):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        output, fourcc, fps, (resolution, resolution))

    total_steps = int(time_in_seconds / string.get_timestep())
    draw_every = int(1.0 / fps / string.get_timestep())
    iteration = tqdm(range(total_steps)) if verbose else range(total_steps)

    for i in iteration:
        if i % draw_every == 0:
            r = np.uint8(string.get_render(resolution))
            r = np.flip(r, axis=0)
            r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR) * 255
            writer.write(r)

        string.step()

    writer.release()


def video_sample_to_plot(sample, spring_constant=None, title=None):
    fig, ax = plt.subplots(1, 5, sharey=True, figsize=(10, 2), dpi=150)
    idx = 0

    if title is not None:
        ax[0].set_title(title)
    elif spring_constant is not None:
        ax[0].set_title(f"Spring Constant: {spring_constant:.2f}")

    total_steps = sample.shape[0]
    draw_every = (total_steps) // 5
    for i in range(total_steps):
        if i % draw_every == 0 and idx < 5:
            render = np.flip(np.float32(sample[i]))
            ax[idx].imshow(render, cmap='gray', vmin=0, vmax=1)
            idx += 1
