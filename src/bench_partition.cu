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

        if(mode == "cpu"){
            double t0 = now_ms();
            partition_cpu(input, output, counts, offsets, bits);
            double t1 = now_ms();
            time_ms = t1 - t0;
        } else if (mode == "gpu"){
            std::cerr << "GPU mode not yet implemented\n";
            return 1;
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
            << throughput_mtuples_s
            << "\n";
    }

    return 0;
}
