#include <cstddef>

namespace forgecc {

void gpu_malloc(void** ptr, size_t bytes) {
    *ptr = nullptr;
}

void gpu_free(void* ptr) {
}

void gpu_memcpy_to_device(void* dst, const void* src, size_t bytes) {
}

void gpu_memcpy_to_host(void* dst, const void* src, size_t bytes) {
}

void gpu_launch_matmul(const float*, const float*, float*, int, int, int) {
}

}
