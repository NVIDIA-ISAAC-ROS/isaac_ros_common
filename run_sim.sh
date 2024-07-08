#!/bin/bash

tmux has-session -t agipix_sim
if [ $? == 0 ]; then
    tmux kill-session -t agipix_sim
fi
tmux new-session -s agipix_sim -n main -d # new session
tmux new-window -n debug -t agipix_sim # new window

tmux select-window -t main

# divide
tmux split-window -h -t 1
tmux split-window -v -t 1

# run
sleep 1
tmux send-keys -t 3 "cd ~/PegasusSimulator/examples" C-m
tmux send-keys -t 3 "ISAACSIM_PYTHON 8_agipix.py" 

tmux send-keys -t 2 "cd && ./QGroundControl.AppImage" C-m

tmux send-keys -t 1 "" C-m

#tmux send-keys -t 2 "cd /workspaces/px4_ros2" C-m

#debug
tmux select-window -t debug

tmux select-window -t main
#----------------------------------------------------------------------------
tmux attach -t agipix_sim # needed to run
