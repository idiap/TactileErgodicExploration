This package implements an ergodic controller on point clouds and it is the supplementary material of the paper "Tactile Ergodic Coverage on Curved Surfaces"

Link to the paper: http://arxiv.org/abs/2402.04862

[[Google colab version]](https://colab.research.google.com/drive/1WFfS4oKQ089wCDgMpZatVTfI6JyIskCY?usp=sharing)

<img src="data/gifs/bunny.gif" alt="drawing" width="500"/>
<img src="data/gifs/cup_plate.gif" alt="drawing" width="500"/>




Paper webpage including interactive plots and real-world experiment videos:

https://sites.google.com/view/tactile-ergodic-control/


## Table of Contents

Notebooks:
- tactile_ergodic_control.ipynb
    - A notebook describing the whole method from start to finish with references and equations.

- Google colab version you can play with this without installing anything to your computer: 
  - https://colab.research.google.com/drive/1WFfS4oKQ089wCDgMpZatVTfI6JyIskCY?usp=sharing


Utilities:
- pointcloud_utils.py
    - Point cloud operations such as read/write kNN queries, gradient computation, etc.
- plotting_utils.py
    - A collection of plotting utility functions used by the notebooks.
- virtual_agents.py
    - Classes for the first and second order virtual agents.

Point clouds:
- Stanford Bunny from the original dataset bun270.ply, 'X' image projected to it using set_point_cloud_target.ipynb 
- A random cup we found in the office and recorded using our setup, 'X' image projected to it using set_point_cloud_target.ipynb 
- A plate from IKEA, recorded using our setup (it includes the exploration target by itself)


## Dependencies

To compute the discrete Laplacian on point clouds (and also meshes if you want)
robust_laplacian: https://github.com/nmwsharp/robust-laplacians-py
```
@article{Sharp:2020:LNT,
  author={Nicholas Sharp and Keenan Crane},
  title={{A Laplacian for Nonmanifold Triangle Meshes}},
  journal={Computer Graphics Forum (SGP)},
  volume={39},
  number={5},
  year={2020}
}
```

For geometric algebra operations:
pygafro: https://gitlab.idiap.ch/tloew/gafro
```
@article{loewGeometricAlgebraOptimal2023,
  title = {Geometric {{Algebra}} for {{Optimal Control}} with {{Applications}} in {{Manipulation Tasks}}},
  author = {L\"ow, Tobias and Calinon, Sylvain},
  date = {2023},
  journal = {IEEE Transactions on Robotics},
  doi = {10.1109/TRO.2023.3277282}
}
```

For basic point cloud operations (another library can be easily used instead):
open3d: https://www.open3d.org

For plotting and point cloud visualizations:
plotly: https://plotly.com

For sparse matrix operations:
scipy: https://scipy.org

For linear algebra operations:
numpy: https://numpy.org


# Copyright and License

Please see the LICENSE for more information.

Contact: cem.bilaloglu@epfl.ch