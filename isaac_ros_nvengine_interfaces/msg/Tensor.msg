# Name of the tensor
string name

# Shape information for tensor
TensorShape shape

# Data type for tensor
# Use the following values to represent these data types:
# - 1: "uint8"
# - 2: "int8"
# - 3: "uint16"
# - 4: "int16"
# - 5: "uint32"
# - 6: "int32"
# - 7: "uint64"
# - 8: "int64"
# - 9: "float32"
# - 10: "float64"
int32 data_type

# Strides of tensor (byte size for each dimension)
uint64[] strides

# Data buffer
uint8[] data