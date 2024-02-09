Bag files of the real-world experiments are in the ```bags``` folder.


First ensure that you use the time published by the bag file for all the following steps. Otherwise you will get an error from the TF.

```
rosparam set use_sim_time true
```

You can select the bagfile and set parameters for the playback. It'll wait for user to press space for the playback to begin.

```
rosbag play --clock --pause -r <playback_speed_multiplier>  --start=<start_time> <bagfile_path>
```

## Start times for different bagfiles

- plate_multi_target: 18
- bowl_RLI: 28
- cup_X: 84
- plate_shapes: 27

## Getting the point cloud  out of the bagfiles

```
rosrun pcl_ros bag_to_pcd <bagfile_path> <topic_name> <output_dir>
```
Topics that you can retrieve

- /goal_pc
- /heat
- /coverage

