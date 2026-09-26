#pragma once
#include <string>

class Driver {
public:
    int run(int argc, char* argv[]);

private:
    std::string inputFile;
    std::string outputFile;
    bool emitIR      = false;
    bool emitForgeIR = false;

    void parseArgs(int argc, char* argv[]);
    void compile();
};