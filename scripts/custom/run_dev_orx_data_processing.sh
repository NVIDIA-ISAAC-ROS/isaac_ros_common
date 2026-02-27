./../run_dev.sh -i ros2_humble.user.orx_data_processing -a "--user admin \
    -v /home/${USER}/dev/orx/data/experiment_config/rosbag_recorder_config:/home/admin/config \
    -v /home/${USER}/dev/orx/data/rosbag_recordings:/home/admin/data/rosbag_recordings"