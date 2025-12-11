# Correlated timestamp
# Nova has three seperate clocks that we must keep track of, the PTP hardware clock (PHC), the
# Timer's System Counter (TSC), and the system clock mainatined by the kernel (sys).
# This struct contains two pairs of correlated timestamps, phc_val_ anc tsc_val are taken
# simultanously, and phc2_val and sys_val_ are taken simultanously.
# phc and tsc are captured at the same time

std_msgs/Header header

uint64 phc_val  # PTP Hardware Clock (PHC)
uint64 tsc_val  # Timer's System Counter (TSC)

# phc2 and tsc are captures at the same time.
uint64 phc2_val  # PTP Hardware Clock (PHC), same clock as phc_val_ above, but taken at a
                 # different time
uint64 sys_val   # System Clock (Sys)

# Deprecated, previously used for sanity checking output from NVPPS, but we no longer use NVPPS
uint64 phc_latency
