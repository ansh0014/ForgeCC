#pragma once
#include <string>

namespace forgecc {

enum class Device { CPU, GPU };

inline std::string deviceName(Device d) {
    return (d == Device::GPU) ? "GPU" : "CPU";
}

}
