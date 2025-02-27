
# Download spryTrack SDK
wget --user ftkuser --password 059_ac9c9e270 https://atracsys.com/_support/controlledAccess/spryTrack/4_9_0/1554-spryTrack_v4.9.0_x64.tar.xz
tar -xJvf 1554-spryTrack_v4.9.0_x64.tar.xz
rm 1554-spryTrack_v4.9.0_x64.tar.xz
cd spryTrack_SDK-v4.9.0-linux64

# Set system path in bashrc
echo "export ATRACSYS_SDK_HOME=$(pwd)" >> ~/.bashrc
source ~/.bashrc

# Create udev daemon rule
sudo sh -c "echo 'SUBSYSTEM==\"usb\", ATTR{idVendor}==\"1c82\", ATTR{idProduct}==\"0200\", MODE=\"0666\"' > /etc/udev/rules.d/51-atracsys-stk.rules"
sudo udevadm control --reload-rules