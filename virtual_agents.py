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

import numpy as np


class FirstOrderAgent:
    """
    A class representing a first-order agent.

    Attributes:
        x (numpy.ndarray): The position of the agent.
        dx (numpy.ndarray): The velocity of the agent.
        max_velocity (float): The maximum velocity of the agent.
        dt (float): The time step for updating the agent's position.
        dim_t (int): The dimension of time for storing the agent's position history.
        x_arr (numpy.ndarray): The array for storing the agent's position history.
        t (int): The current time step for storing the agent's position history.

    Methods:
        update(gradient): Updates the agent's velocity and position based on the given gradient.
    """

    def __init__(self, x, max_velocity=3, dt=1, dim_t=None):
        self.x = np.array(x)  # position
        self.dx = np.zeros(self.x.shape)
        self.max_velocity = max_velocity
        self.dt = dt
        self.dim_t = dim_t
        if self.dim_t is None:
            self.x_list = []
        else:
            self.x_arr = np.zeros((self.dim_t, 3))
            self.t = 0

    def update(self, gradient):
        """
        Updates the agent's velocity and position based on the given gradient.

        Args:
            gradient (numpy.ndarray): The gradient used to update the agent's velocity.

        Returns:
            None
        """
        self.dx = gradient  # update the vel.
        if np.linalg.norm(gradient) > self.max_velocity:
            # clamp the vel.
            self.dx = self.max_velocity * gradient / np.linalg.norm(gradient)
        self.x = self.x + self.dt * self.dx

        if self.dim_t is None:
            self.x_list.append(self.x)
        else:
            self.x_arr[self.t, :] = self.x
            self.t += 1

    def __str__(self):
        return f"dx:{np.linalg.norm(self.dx)}"


class SecondOrderAgent:
    """
    A class representing a second-order agent.

    Attributes:
        x (numpy.ndarray): The position of the agent.
        dx (numpy.ndarray): The velocity of the agent.
        ddx (numpy.ndarray): The acceleration of the agent.
        dt (float): The time step for updating the agent's state.
        dim_t (int): The dimension of time (optional).
        max_velocity (float): The maximum velocity of the agent.
        max_acceleration (float): The maximum acceleration of the agent.

    Methods:
        update(gradient): Updates the agent's state based on the given gradient.
    """

    def __init__(self, x, max_velocity=3, max_acceleration=1, dt=1, dim_t=None):
        self.x = np.array(x)  # position
        self.dx = np.zeros(self.x.shape)
        self.ddx = np.zeros(self.x.shape)
        self.dt = dt
        self.dim_t = dim_t
        if self.dim_t is not None:
            self.x_arr = np.zeros((dim_t, 3))
            self.t = 0

        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def update(self, gradient):
        """
        Updates the agent's state based on the given gradient.

        Args:
            gradient (numpy.ndarray): The gradient used to update the acceleration.

        Returns:
            None
        """
        ddx = gradient  # update the acc.
        if np.linalg.norm(gradient) > self.max_acceleration:
            # clamp acc.
            ddx = self.max_acceleration * gradient / np.linalg.norm(gradient)
        # print(f"ddx:{ddx}")
        x_prev = np.copy(self.x)
        self.x += +self.dt * self.dx + 0.5 * self.dt * self.dt * ddx
        # print(f"x:{self.x-x_prev}")
        # self.x += 0
        if self.dim_t is not None:
            self.x_arr[self.t, :] = self.x
            self.t += 1
        self.dx += self.dt * ddx  # update the vel.
        if np.linalg.norm(self.dx) > self.max_velocity:
            # clamp the vel.
            self.dx = self.max_velocity * self.dx / np.linalg.norm(self.dx)

    def __str__(self):
        return f"dx:{np.linalg.norm(self.dx)} ddx:{np.linalg.norm(self.ddx)}"
