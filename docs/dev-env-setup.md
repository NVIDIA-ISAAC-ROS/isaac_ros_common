# Isaac ROS Development Environment Setup

## Quickstart
> **Note:** Before you begin, verify that you have sufficient storage space available on your device. We recommend at least **30 GB**, to account for the size of the container and datasets.
> 
> On Jetson Xavier platforms, an external drive is **required** to have enough storage space.


1. Configure `nvidia-container-runtime` as the default runtime for Docker. 
   
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
2. Then, restart Docker: 
   ```bash
   sudo systemctl daemon-reload && sudo systemctl restart docker
   ```
3. Install [Git LFS](https://git-lfs.github.com/) in order to pull down all large files:  
    ```bash
    sudo apt-get install git-lfs
    ```  
4. Finally, create a ROS2 workspace for experimenting with Isaac ROS:  
    ```bash
    mkdir -p  ~/workspaces/isaac_ros-dev/src
    ``` 
