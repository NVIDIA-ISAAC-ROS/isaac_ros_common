std_msgs/Header header 
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step

# Filled with PID and file descriptor(exported from GPU memory)
int32[] data # (PID, FD)
uint8[] cuda_event_handle
string uid
uint32 device_id