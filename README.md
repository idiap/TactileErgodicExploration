A Python package for ergodic control on point cloud using diffusion. It is supplementary material for the paper "Tactile Ergodic Control Using Diffusion and Geometric Algebra." The package uses Laplacian eigenbasis for computing the potential field resulting from the diffusion on the point cloud. Then, it uses the heat-equation-driven area coverage (HEDAC) method to guide the exploration agents for tactile ergodic control tasks. This research is conducted at the Robot Learning and Interaction group of the Idiap Research Institute.

Link to the paper: http://arxiv.org/abs/2402.04862
```
@online{bilalogluTactileErgodicControl2024,
  title = {Tactile {{Ergodic Control Using Diffusion}} and {{Geometric Algebra}}},
  author = {Bilaloglu, Cem and Löw, Tobias and Calinon, Sylvain},
  date = {2024-02-07},
  eprint = {2402.04862},
  eprinttype = {arxiv},
  eprintclass = {cs},
  url = {http://arxiv.org/abs/2402.04862}}
```

Paper webpage including interactive plots and real-world experiment videos:

https://geometric-algebra.tobiloew.ch/tactile_ergodic_control/

The real-world experiment data stored as bag files are in the ```bags``` folder.

It is possible to generate all the simulation data and the plots used in the paper using this repository. 

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

Notebooks:
- set_point_cloud_target.ipynb  
    - Creates point clouds with exploration targets. Loads a user-defined point cloud, projects a user-defined target, visualizes it and saves the point cloud for further use with the method.
- laplacian_hedac_point_cloud.ipynb
    - A notebook describing the whole method from start to finish with references and equations. All the simulation data used in the paper are generated using the "multi" option for the simulation_type and running the notebook once for each shape.
- process_results.ipynb
    - A notebook for processing the experiment data and for generating the plots. All the plots used in the paper are generated using this notebook.


Scripts:
- hedac_utils.py
    - A collection of utility functions related to the method.
- plotting_utils.py
    - A collection of plotting utility functions used by the notebooks.
- virtual_agents.py
    - Classes for the first and second order virtual agents.

Point clouds:
- Stanford Bunny from the original dataset bun270.ply, 'X' image projected to it using set_point_cloud_target.ipynb 
- A random cup we found in the office and recorded using our setup, 'X' image projected to it using set_point_cloud_target.ipynb 
- A plate from IKEA, recorded using our setup (it includes the exploration target by itself)


If you use this repository for an academic use, you can cite our paper 

```
TODO
```