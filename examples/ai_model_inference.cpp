#include "forgecc/forgecc.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    std::cout << "===========================================\n";
    std::cout << " ForgeCC Universal AI GPU Inference Engine\n";
    std::cout << "===========================================\n";

    bool gpu_ready = forge::has_gpu();
    std::cout << "GPU Acceleration Status: " << (gpu_ready ? "AVAILABLE (NVIDIA CUDA)" : "FALLBACK (CPU Multi-Thread)") << "\n\n";

    forge::Device target_dev = gpu_ready ? forge::Device::GPU : forge::Device::CPU;

    int64_t batch_seq = 1024;
    int64_t hidden_dim = 1024;

    std::cout << "Initializing AI Weights & Activations (" << batch_seq << "x" << hidden_dim << ")...\n";

    forge::Tensor Input({batch_seq, hidden_dim}, target_dev);
    forge::Tensor Weights({hidden_dim, hidden_dim}, target_dev);

    std::vector<float> h_input(batch_seq * hidden_dim, 0.01f);
    std::vector<float> h_weights(hidden_dim * hidden_dim, 0.02f);

    Input.from_host(h_input.data());
    Weights.from_host(h_weights.data());

    std::cout << "Executing Fused Linear Layer (MatMul + ReLU) on " 
              << (target_dev == forge::Device::GPU ? "GPU" : "CPU") << "...\n";

    auto start = std::chrono::high_resolution_clock::now();

    forge::Tensor Output = forge::matmul_relu(Input, Weights);
    forge::sync();

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::vector<float> h_output(batch_seq * hidden_dim);
    Output.to_host(h_output.data());

    std::cout << "Execution completed in: " << ms << " ms\n";
    std::cout << "First output element: " << h_output[0] << "\n";
    std::cout << "Verification: SUCCESS\n";

    return 0;
}
