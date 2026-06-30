import numpy as np
import torch
import matplotlib.pyplot as plt
import PIL
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy import ndimage
from skimage.measure import marching_cubes

from models import Generator

def render_orthographic_faces(fake_images: torch.Tensor, level: float=0.0) -> PIL.Image.Image:
    fig = plt.figure(figsize=(12, 8))
    verts, faces, normals, values = marching_cubes(
        fake_images,
        level=level,
    )

    views = [
        (90, 0),
        (-90, 0),
        (0, 0),
        (0, 180),
        (0, 90),
        (0, -90),
    ]

    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")

        mesh = Poly3DCollection(
            verts[faces],
            alpha=0.9,
            linewidth=0.0,
        )
        mesh.set_facecolor([1.0, 0.0, 0.0, 0.9])

        ax.add_collection3d(mesh)

        ax.set_box_aspect([1, 1, 1])

        ax.set_xlim(0, fake_images.shape[0])
        ax.set_ylim(0, fake_images.shape[1])
        ax.set_zlim(0, fake_images.shape[2])

        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

    fig.subplots_adjust(
        left=0,
        right=1,
        top=0.95,
        bottom=0,
        hspace=0.05,
        wspace=0.05,
    )

    fig.canvas.draw()

    image = Image.fromarray(
        np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    )
    plt.close(fig)
    return image

def render_path(generator: Generator, noise: torch.Tensor,elev: int=30) -> np.ndarray:
    with torch.no_grad():
        fake_images = generator(noise).squeeze().cpu().numpy()

    level = 0.0
    min_val = float(fake_images.min())
    max_val = float(fake_images.max())

    if not (min_val < level < max_val):
        level = (min_val + max_val) / 2.0

    occupancy = fake_images > level

    labeled, num_features = ndimage.label(occupancy)

    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0

        largest_label = component_sizes.argmax()
        largest_component = labeled == largest_label

        fake_images = np.where(largest_component, fake_images, min_val)

    verts, faces, normals, values = marching_cubes(
        fake_images,
        level=level,
    )

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")

    mesh = Poly3DCollection(
        verts[faces],
        alpha=0.9,
        linewidth=0.0,
    )
    mesh.set_facecolor([1.0, 0.0, 0.0, 0.9])
    ax.add_collection3d(mesh)

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlim(0, fake_images.shape[0])
    ax.set_ylim(0, fake_images.shape[1])
    ax.set_zlim(0, fake_images.shape[2])

    ax.set_axis_off()

    frames = []
    for azim in np.linspace(0, 360, 120, endpoint=False):
        ax.view_init(elev=elev, azim=azim)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame.copy())

    plt.close(fig)

    return np.stack(frames)