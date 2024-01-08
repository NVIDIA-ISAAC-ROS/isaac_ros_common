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

# Latency of reading the PHC, slightly subtle, as this only applies to PHC2
# Basically the simultanous reading of sys<->phc2 within nvpps does not account
# The time it takes to read from the PHC, which can be several microseconds.
# The value of phc2 published by the correlated_timestamp_driver already accounts for
# This value, and phc_latency is provided to expose this value to GXF for sanity checking
uint64 phc_latency
