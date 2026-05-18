NVCC = nvcc
CXXFLAGS = -O3 -std=c++17 -arch=sm_86 # RTX 3080 is Ampere

TARGET = bench_partition
SRC = src/bench_partition.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)