#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct tuple_t {
    uint64_t key;
    uint64_t payload;
};

static inline void check_cuda(
    cudaError_t res,
    const char *call,
    const char *file,
    int line
){
    if (res != cudaSuccess){
        std::cerr
            << "CUDA error at " << file << ":" << line << "\n"
            << "Call: " << call << "\n"
            << "Error: " << cudaGetErrorString(res) << "\n";
        std::exit(1);
    }
}

#define CHECK_CUDA(call) \
    do { \
        check_cuda((call), #call, __FILE__, __LINE__); \
    } while (0)

struct gpu_timing_t {
    double alloc_ms = 0.0;
    double h2d_input_ms = 0.0;
    double count_kernel_ms = 0.0;
    double d2h_counts_ms = 0.0;
    double cpu_prefix_ms = 0.0;
    double h2d_offsets_ms = 0.0;
    double scatter_kernel_ms = 0.0;
    double d2h_output_ms = 0.0;
    double free_ms = 0.0;
};


static inline uint32_t get_partition(uint64_t key, int bits){
    return key & ((1u << bits) - 1u);
}

static double now_ms(){
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()
    ).count();
}

void generate_input(
    std::vector<tuple_t> &input,
    const std::string &distribution,
    int bits
) {
    std::mt19937_64 rng(12345);
    uint64_t partition_mask = (1ull << bits) - 1ull;

    for(size_t i = 0; i < input.size(); i++){
        uint64_t key;

        if (distribution == "uniform"){
            key = rng();
        } else if (distribution == "skewed"){
            // ~80% of tuples go to partition 0
            if ((rng() % 100) < 80){
                key = rng() & ~partition_mask;
            } else {
                key = rng();
            }
        } else {
            std::cerr << "Unknown distribution: " << distribution << "\n";
            std::exit(1);
        }

        input[i].key = key;
        input[i].payload = i;
    }
}

void partition_cpu(
    const std::vector<tuple_t> &input,
    std::vector<tuple_t> &output,
    std::vector<uint32_t> &counts,
    std::vector<uint32_t> &offsets,
    int bits
) {
    size_t n = input.size();
    uint32_t partitions = 1u << bits;

    std::fill(counts.begin(), counts.end(), 0);
    std::fill(offsets.begin(), offsets.end(), 0);

    // pass 1: count partition sizes
    for(size_t i = 0; i < n; i++){
        uint32_t p = get_partition(input[i].key, bits);
        counts[p]++;
    }

    // pass 2: exclusive prefix sum
    offsets[0] = 0;
    for(uint32_t p = 1; p < partitions; p++){
        offsets[p] = offsets[p - 1] + counts[p - 1];
    }

    // pass 3: scatter tuples into output
    std::vector<uint32_t> write_pos(partitions, 0);

    for(size_t i = 0; i < n; i++){
        uint32_t p = get_partition(input[i].key, bits);
        uint32_t pos = offsets[p] + write_pos[p]++;
        output[pos] = input[i];
    }
}

__global__ void count_kernel(const tuple_t *input, size_t n, int bits, uint32_t *counts){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        uint32_t mask = (1u << bits) - 1u;
        uint32_t p = input[idx].key & mask;

        atomicAdd(&counts[p], 1);
    }
}

__global__ void scatter_kernel(
    const tuple_t *input, 
    tuple_t *output, 
    size_t n, 
    int bits, 
    const uint32_t *offsets,
    uint32_t *write_pos
){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n){
        uint32_t mask = (1u << bits) - 1u;
        uint32_t p = input[idx].key & mask;
        uint32_t local_pos = atomicAdd(&write_pos[p], 1);
        uint32_t out_pos = offsets[p] + local_pos;

        output[out_pos] = input[idx]; 
    }
}

double partition_gpu(
    const std::vector<tuple_t> &input,
    std::vector<tuple_t> &output,
    std::vector<uint32_t> &counts,
    std::vector<uint32_t> &offsets,
    int bits,
    gpu_timing_t &timing
) {
    size_t n = input.size();
    uint32_t partitions = 1u << bits;

    tuple_t *d_input = nullptr;
    tuple_t *d_output = nullptr;
    uint32_t *d_counts = nullptr;
    uint32_t *d_offsets = nullptr;
    uint32_t *d_write_pos = nullptr;

    size_t tuple_bytes = n * sizeof(tuple_t);
    size_t partition_bytes = partitions * sizeof(uint32_t);

    double t_total = now_ms();
    double t_phase = now_ms();

    CHECK_CUDA(cudaMalloc((void **)&d_input, tuple_bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_output, tuple_bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_counts, partition_bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_offsets, partition_bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_write_pos, partition_bytes));

    timing.alloc_ms = now_ms() - t_phase;
    t_phase = now_ms();

    CHECK_CUDA(cudaMemcpy(d_input, input.data(), tuple_bytes, cudaMemcpyHostToDevice));

    timing.h2d_input_ms = now_ms() - t_phase;
    t_phase = now_ms();

    CHECK_CUDA(cudaMemset(d_counts, 0, partition_bytes));

    int threads_per_block = 256;
    int blocks = static_cast<int>((n + threads_per_block - 1) / threads_per_block);

    // pass 1: count partition sizes on GPU
    count_kernel<<<blocks, threads_per_block>>>(d_input, n, bits, d_counts);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    timing.count_kernel_ms = now_ms() - t_phase;
    t_phase = now_ms();

    CHECK_CUDA(cudaMemcpy(counts.data(), d_counts, partition_bytes, cudaMemcpyDeviceToHost));

    timing.d2h_counts_ms = now_ms() - t_phase;
    t_phase = now_ms();

    offsets[0] = 0;
    for(uint32_t p = 1; p < partitions; p++){ offsets[p] = offsets[p - 1] + counts[p - 1]; }

    // sanity check
    uint32_t total = offsets[partitions - 1] + counts[partitions - 1];
    if(total != n){ std::cerr
                        << "GPU prefix sum error: total=" << total
                        << ", expected n=" << n << "\n";
                    std::exit(1); }

    timing.cpu_prefix_ms = now_ms() - t_phase;
    t_phase = now_ms();

    // copy offsets to GPU
    CHECK_CUDA(cudaMemcpy(d_offsets, offsets.data(), partition_bytes, cudaMemcpyHostToDevice));

    timing.h2d_offsets_ms = now_ms() - t_phase;
    t_phase = now_ms();

    // clear per-partition write positions
    CHECK_CUDA(cudaMemset(d_write_pos, 0, partition_bytes));

    // pass 2: scatter tuples into partitioned output
    scatter_kernel<<<blocks, threads_per_block>>>(d_input, d_output, n, bits, d_offsets, d_write_pos);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    timing.scatter_kernel_ms = now_ms() - t_phase;
    t_phase = now_ms();

    // copy partitioned output back to cpu mem
    CHECK_CUDA(cudaMemcpy(output.data(), d_output, tuple_bytes, cudaMemcpyDeviceToHost));

    timing.d2h_output_ms = now_ms() - t_phase;
    t_phase = now_ms();

    // free GPU mem
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_write_pos));

    timing.free_ms = now_ms() - t_phase;

    return now_ms() - t_total;
}

bool validate_partitioning(
    const std::vector<tuple_t> &output,
    const std::vector<uint32_t> &counts,
    const std::vector<uint32_t> &offsets,
    int bits
){
    uint32_t partitions = 1u << bits;

    for(uint32_t p = 0; p < partitions; p++){
        uint32_t start = offsets[p];
        uint32_t end = start + counts[p];

        for(uint32_t i = start; i < end; i++){
            uint32_t actual = get_partition(output[i].key, bits);

            if(actual != p){
                std::cerr
                    << "Validation failed at output[" << i << "]. "
                    << "Expected partition " << p
                    << ", got " << actual << "\n";
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char **argv){
    if(argc != 6){
        std::cerr
            << "Usage: " << argv[0]
            << " <cpu|gpu> <n> <bits> <repeats> <uniform|skewed>\n";
        return 1;
    }

    std::string mode = argv[1];
    size_t n = std::stoull(argv[2]);
    int bits = std::stoi(argv[3]);
    int repeats = std::stoi(argv[4]);
    std::string distribution = argv[5];

    if(bits <= 0 || bits > 20){
        std::cerr << "bits must be between 1 and 20\n";
        return 1;
    }

    uint32_t partitions = 1u << bits;

    std::vector<tuple_t> input(n);
    std::vector<tuple_t> output(n);
    std::vector<uint32_t> counts(partitions);
    std::vector<uint32_t> offsets(partitions);

    generate_input(input, distribution, bits);

    for(int r = 0; r < repeats; r++){
        double time_ms = 0.0;
        gpu_timing_t gpu_timing;

        if(mode == "cpu"){
            double t0 = now_ms();
            partition_cpu(input, output, counts, offsets, bits);
            double t1 = now_ms();
            time_ms = t1 - t0;

        } else if (mode == "gpu"){
            time_ms = partition_gpu(input, output, counts, offsets, bits, gpu_timing);

        } else {
            std::cerr << "Unknown mode: " << mode << "\n";
            return 1;
        }

        if(!validate_partitioning(output, counts, offsets, bits)){
            return 1;
        }

        double throughput_mtuples_s = ((double)n / 1e6) / (time_ms / 1000.0);

        std::cout
            << mode << ","
            << n << ","
            << bits << ","
            << partitions << ","
            << distribution << ","
            << r << ","
            << time_ms << ","
            << throughput_mtuples_s << ","
            << gpu_timing.alloc_ms << ","
            << gpu_timing.h2d_input_ms << ","
            << gpu_timing.count_kernel_ms << ","
            << gpu_timing.d2h_counts_ms << ","
            << gpu_timing.cpu_prefix_ms << ","
            << gpu_timing.h2d_offsets_ms << ","
            << gpu_timing.scatter_kernel_ms << ","
            << gpu_timing.d2h_output_ms << ","
            << gpu_timing.free_ms
            << "\n";
    }

    return 0;
}
