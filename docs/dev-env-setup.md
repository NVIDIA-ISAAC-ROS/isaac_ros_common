# Isaac ROS Development Environment Setup

## Quickstart
> **Note:** Before you begin, verify that you have sufficient storage space available on your device. We recommend at least **30 GB**, to account for the size of the container and datasets.
> 
> On Jetson Xavier platforms, an external drive is **required** to have enough storage space.

1. First, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html):
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && \
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
        curl -s -L https://nvidia.github.io/libnvidia-container/experimental/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

    ```bash
    sudo apt-get update && \
        sudo apt-get install nvidia-container-toolkit=1.10.0~rc.3-1
    ```

2. Next, configure `nvidia-container-runtime` as the default runtime for Docker. 
   
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
5. Finally, create a ROS2 workspace for experimenting with Isaac ROS:  
    ```bash
    mkdir -p  ~/workspaces/isaac_ros-dev/src
    ``` 
