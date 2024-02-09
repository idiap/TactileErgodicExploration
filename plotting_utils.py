"""
    Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
    Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

    This file is part of tactileErgodicExploration.

    tactileErgodicExploration is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    tactileErgodicExploration is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with tactileErgodicExploration. If not, see <http://www.gnu.org/licenses/>.
"""

import plotly.graph_objects as go
import chart_studio

import open3d as o3d

import chart_studio

# Set up credentials for chart_studio if you want to upload the plots to the cloud
chart_studio.tools.set_credentials_file(
    username="cembilaloglu", api_key="T5GCkRWnXddyei8Qac3W"
)

import numpy as np


def show_plot(plots):
    layout = go.Layout(scene=dict(aspectmode="data"))
    fig = go.Figure(data=plots, layout=layout)
    camera_params = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.0, y=0.0, z=2.0),
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        scene_camera=camera_params,
    )
    return fig


def visualize_trajectory(x_arr, plots=None, is_show_plot=True):
    """
    Visualizes a point cloud in a 3D scatter plot.

    Args:
        vertices (numpy.ndarray): The vertices of the point cloud.
        colors (numpy.ndarray, optional): The colors of the points. Defaults to None.
        plots (list, optional): The existing plots to be updated. Defaults to None.
        point_size (int, optional): The size of the points in the scatter plot. Defaults to 2.
    """
    if plots is None:
        plots = []
    initial_position_plot = go.Scatter3d(
        x=[x_arr[0, 0]],
        y=[x_arr[0, 1]],
        z=[x_arr[0, 2]],
        name="Agent Initial Position",
        marker=dict(
            size=10,
            color="green",
        ),
    )
    plots.append(initial_position_plot)
    trajectory_plot = go.Scatter3d(
        x=x_arr[:, 0],
        y=x_arr[:, 1],
        z=x_arr[:, 2],
        mode="lines",  # Change mode to "lines"
        name="Agent Trajectory",
        line=dict(width=8, color="black"),
    )
    plots.append(trajectory_plot)
    final_position_plot = go.Scatter3d(
        x=[x_arr[-1, 0]],
        y=[x_arr[-1, 1]],
        z=[x_arr[-1, 2]],
        name="Agent Final Position",
        marker=dict(
            size=10,
            color="magenta",
        ),
    )
    plots.append(final_position_plot)

    if is_show_plot:
        return show_plot(plots)
    else:
        return plots


def visualize_point_cloud(
    vertices, colors=None, plots=None, is_show_plot=True, point_size=2
):
    """
    Visualizes a point cloud in a 3D scatter plot.

    Args:
        vertices (numpy.ndarray): The vertices of the point cloud.
        colors (numpy.ndarray, optional): The colors of the points. Defaults to None.
        plots (list, optional): The existing plots to be updated. Defaults to None.
        point_size (int, optional): The size of the points in the scatter plot. Defaults to 2.
    """
    if plots is None:
        plots = []
    if colors is None:
        colors = np.zeros((len(vertices), 1))

    # first construct the open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    color = np.zeros((len(vertices), 3))
    color[:, 0] = colors
    pcd.colors = o3d.utility.Vector3dVector(color)
    # here we downsample the point cloud to have a smaller plot to upload to plotly
    pcd = pcd.voxel_down_sample(voxel_size=0.004)  # downsample
    vertices = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    colors = colors[:, 0]

    scatter = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode="markers",
        name="Point Cloud",
        marker=dict(
            size=point_size,
            color=colors,
            colorscale="bluered",  # You can choose other color scales
            opacity=0.8,
        ),
    )
    plots.append(scatter)
    if is_show_plot:
        return show_plot(plots)

    else:
        return plots
