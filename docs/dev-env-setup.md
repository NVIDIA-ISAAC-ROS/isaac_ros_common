# Isaac ROS Development Environment Setup

## Quickstart

> **Note**: Before you begin, verify that you have sufficient storage space available on your device. We recommend at least **30 GB**, to account for the size of the container and datasets.
>
> On Jetson platforms, an NVMe SSD storage is **required** to have sufficient and fast storage.

1. **On x86_64 platforms**: Install the `nvidia-container-runtime` using the instructions [here](https://github.com/NVIDIA/nvidia-container-runtime#installation).

   **On Jetson platforms**: Follow [this instruction](dev-env-setup_jetson.md) to first set your Jetson up with SSD, then come back to this document and resume from Step 3.

2. Configure `nvidia-container-runtime` as the default runtime for Docker.

   Using your text editor of choice, add the following items to `/etc/docker/daemon.json`.

    ```json
    {
        ...
        "runtimes": {
            "nvidia": {
                "path": "nvidia-container-runtime",
                "runtimeArgs": []
            }
        },
        "default-runtime": "nvidia"
        ...
    }
    ```

3. Then, restart Docker:

   ```bash
   sudo systemctl daemon-reload && sudo systemctl restart docker
   ```

4. Install [Git LFS](https://git-lfs.github.com/) in order to pull down all large files:  

    ```bash
    sudo apt-get install git-lfs
    ```  

    ```bash
    git lfs install --skip-repo
    ```

5. Finally, create a ROS 2 workspace for experimenting with Isaac ROS:  

    > **For Jetson setup with SSD as optional storage**:
    >
    > ```bash
    > mkdir -p  /ssd/workspaces/isaac_ros-dev/src
    > echo "export ISAAC_ROS_WS=/ssd/workspaces/isaac_ros-dev/" >> ~/.bashrc
    > source ~/.bashrc
    > ```

    ```bash
    mkdir -p  ~/workspaces/isaac_ros-dev/src
    echo "export ISAAC_ROS_WS=${HOME}/workspaces/isaac_ros-dev/" >> ~/.bashrc
    source ~/.bashrc
    ```

    Note that we are going to use `ISAAC_ROS_WS` environmental variable in the future to refer to this ROS 2 workspace directory.
