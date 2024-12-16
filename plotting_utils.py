"""
    Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
    Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

    This file is part of diffusionVirtualFixtures.

    diffusionVirtualFixtures is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    diffusionVirtualFixtures is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with diffusionVirtualFixtures. If not, see <http://www.gnu.org/licenses/>.
"""

import plotly.graph_objects as go

import open3d as o3d

import numpy as np


def show_plot(plots, camera_params=None, showlegend=True):
    layout = go.Layout(scene=dict(aspectmode="data"))
    fig = go.FigureWidget(data=plots, layout=layout)
    if camera_params is None:
        camera_params = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.0, y=0.0, z=2.0),  # plane
            # eye=dict(x=0.0, y=0.0, z=0.5),  # others
        )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        showlegend=showlegend,
        scene_camera=camera_params,
    )

    # fig.update(layout_coloraxis_showscale=False)
    return fig


def visualize_trajectory(
    x_arr,
    plots=None,
    color="black",
    legendgroup=None,
    showlegend=True,
    experiment_index=0,
    is_show_plot=True,
):
    """
    Visualizes a point cloud in a 3D scatter plot.

    Args:
        vertices (numpy.ndarray): The vertices of the point cloud.
        colors (numpy.ndarray, optional): The colors of the points. Defaults to None.
        plots (list, optional): The existing plots to be updated. Defaults to None.
        point_size (int, optional): The size of the points in the scatter plot.
        Defaults to 2.
    """
    if plots is None:
        plots = []
    initial_position_plot = go.Scatter3d(
        x=[x_arr[0, 0]],
        y=[x_arr[0, 1]],
        z=[x_arr[0, 2]],
        legendgroup=legendgroup,
        showlegend=False,
        name=f"{experiment_index}",
        # legendgroup=legendgroup,
        marker=dict(
            size=8,
            color="green",
        ),
    )
    plots.append(initial_position_plot)
    trajectory_plot = go.Scatter3d(
        x=x_arr[:, 0],
        y=x_arr[:, 1],
        z=x_arr[:, 2],
        mode="lines",  # Change mode to "lines"
        name=f"Agent Trajectory {legendgroup}",
        showlegend=showlegend,
        legendgroup=legendgroup,
        line=dict(width=10, color=color),
        # line=dict(width=8),
    )
    plots.append(trajectory_plot)
    final_position_plot = go.Scatter3d(
        x=[x_arr[-1, 0]],
        y=[x_arr[-1, 1]],
        z=[x_arr[-1, 2]],
        legendgroup=legendgroup,
        showlegend=False,
        marker=dict(
            size=4,
            color="magenta",
        ),
    )
    plots.append(final_position_plot)

    if is_show_plot:
        return show_plot(plots)
    else:
        return plots


def visualize_point_cloud(
    vertices, colors=None, plots=None, is_show_plot=True, point_size=3, voxel_size=None
):
    """
    Visualizes a point cloud in a 3D scatter plot.

    Args:
        vertices (numpy.ndarray): The vertices of the point cloud.
        colors (numpy.ndarray, optional): The colors of the points. Defaults to None.
        plots (list, optional): The existing plots to be updated. Defaults to None.
        point_size (int, optional): The size of the points in the scatter plot.
          Defaults to 2.
    """
    if plots is None:
        plots = []
    if colors is None:
        color_arr = np.zeros((len(vertices), 3))
        colors = color_arr
    elif colors.ndim == 1:
        # print("1D colors")
        color_arr = np.zeros((len(vertices), 3))
        color_arr[:, 0] = colors
    else:
        color_arr = colors

    # first construct the open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(color_arr)
    if voxel_size is not None:
        # here we downsample the point cloud to have a smaller plot to upload to plotly
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)  # downsample
    vertices = np.asarray(pcd.points)
    color_arr = np.asarray(pcd.colors)

    marker = dict(
        size=point_size,
        showscale=True,
        opacity=0.2,
    )
    if colors.ndim == 2:
        # print("3D colors")
        colors = colors.astype(int).astype(str)
        # Create a scatter3d trace with colors
        color_string = [f'rgb({",".join(c)})' for c in colors]
        marker["color"] = color_string

    else:

        marker["color"] = color_arr[:, 0]
        colorscale = "bluered"
        # colorscale="jet"
        # colorscale="rainbow"
        # colorscale = "turbo"
        marker["colorscale"] = colorscale

    scatter = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode="markers",
        name="Point Cloud",
        marker=marker,
    )

    plots.append(scatter)
    if is_show_plot:
        return show_plot(plots)

    else:
        return plots


def visualize_gradient_field(
    vertices,
    gradient_arr,
    plots=None,
    legendgroup=None,
    showlegend=True,
    is_show_plot=True,
    sizeref=1,
):

    # Visualize the gradient field
    # ==============================================================================
    if plots is None:
        plots = []  # we will append the plots to this list]

    # max_val = np.max(ut)
    # scaled_gradient_arr = gradient_arr * (max_val - ut)[:, None]
    # scaled_gradient_arr = gradient_arr

    gradient_field_plot = go.Cone(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        u=gradient_arr[:, 0],
        v=gradient_arr[:, 1],
        w=gradient_arr[:, 2],
        # sizemode="absolute",
        sizeref=sizeref,
    )
    plots.append(gradient_field_plot)
    if is_show_plot:
        return show_plot(plots)

    else:
        return plots


def streamline_plot(
    vertices,
    gradient_arr,
    plots=None,
    legendgroup=None,
    showlegend=True,
    is_show_plot=True,
):

    # Visualize the gradient field
    # ==============================================================================
    if plots is None:
        plots = []  # we will append the plots to this list]

    streamline_plot = go.Streamtube(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        u=gradient_arr[:, 0],
        v=gradient_arr[:, 1],
        w=gradient_arr[:, 2],
    )
    plots.append(streamline_plot)
    if is_show_plot:
        return show_plot(plots)

    else:
        return plots
