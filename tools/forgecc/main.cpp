#include "Driver.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: forgecc <input.cpp>\n";
        return 1;
    }
    Driver driver;
    return driver.run(argc, argv);
}