#include "forgecc_rt/tensor.hpp"
#include <cstdint>
#include <cmath>

extern "C" {

void cpu_matmul(const float* A, const float* B, float* C,
                int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void cpu_relu(const float* in, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
    }
}

void cpu_sigmoid(const float* in, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
}

void cpu_tanh_act(const float* in, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = std::tanh(in[i]);
    }
}

void cpu_gelu(const float* in, float* out, int64_t n) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    for (int64_t i = 0; i < n; ++i) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + std::tanh(k0 * (x + k1 * x * x * x)));
    }
}

void cpu_add(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void cpu_sub(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

void cpu_mul(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void cpu_div(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = (b[i] != 0.0f) ? (a[i] / b[i]) : 0.0f;
    }
}

void cpu_add_bias(const float* in, const float* bias, float* out, int64_t M, int64_t N) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            out[i * N + j] = in[i * N + j] + bias[j];
        }
    }
}

void cpu_sum(const float* in, float* out, int64_t n) {
    float total = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        total += in[i];
    }
    *out = total;
}

void cpu_mean(const float* in, float* out, int64_t n) {
    if (n <= 0) {
        *out = 0.0f;
        return;
    }
    float total = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        total += in[i];
    }
    *out = total / static_cast<float>(n);
}

void cpu_matmul_add(const float* A, const float* B, const float* bias, float* C,
                    int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum + bias[j];
        }
    }
}

void cpu_matmul_add_relu(const float* A, const float* B, const float* bias, float* C,
                         int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            float val = sum + bias[j];
            C[i * N + j] = (val > 0.0f) ? val : 0.0f;
        }
    }
}

void cpu_matmul_add_relu_residual(const float* A, const float* B, const float* bias, const float* residual, float* C,
                                  int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            float val = sum + bias[j];
            float activated = (val > 0.0f) ? val : 0.0f;
            C[i * N + j] = activated + residual[i * N + j];
        }
    }
}

void cpu_add_relu(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float val = a[i] + b[i];
        out[i] = (val > 0.0f) ? val : 0.0f;
    }
}

void cpu_matmul_backward(const float* grad_out, const float* A, const float* B,
                         float* grad_A, float* grad_B,
                         int64_t M, int64_t K, int64_t N) {
    if (grad_A) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t k = 0; k < K; ++k) {
                float sum = 0.0f;
                for (int64_t j = 0; j < N; ++j) {
                    sum += grad_out[i * N + j] * B[k * N + j];
                }
                grad_A[i * K + k] += sum;
            }
        }
    }
    if (grad_B) {
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t i = 0; i < M; ++i) {
                    sum += A[i * K + k] * grad_out[i * N + j];
                }
                grad_B[k * N + j] += sum;
            }
        }
    }
}

void cpu_relu_backward(const float* grad_out, const float* in, float* grad_in, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        grad_in[i] = (in[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
}

void cpu_sgd_step(float* weights, const float* grads, float* velocity,
                  float lr, float momentum, float weight_decay, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float g = grads[i];
        if (weight_decay != 0.0f) {
            g += weight_decay * weights[i];
        }
        if (velocity != nullptr && momentum > 0.0f) {
            velocity[i] = momentum * velocity[i] + g;
            weights[i] -= lr * velocity[i];
        } else {
            weights[i] -= lr * g;
        }
    }
}

void cpu_adam_step(float* weights, const float* grads, float* m, float* v,
                   float lr, float beta1, float beta2, float eps,
                   float weight_decay, int64_t step, int64_t n) {
    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    for (int64_t i = 0; i < n; ++i) {
        float g = grads[i];
        if (weight_decay != 0.0f) {
            g += weight_decay * weights[i];
        }
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);
        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;
        weights[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

}


