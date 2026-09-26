#include "Driver.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: forgecc <input.cpp> [--emit-ir] [--emit-forge-ir] [-o output]\n";
        return 1;
    }
    Driver driver;
    return driver.run(argc, argv);
}