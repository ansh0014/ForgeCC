#pragma once
#include <string>
#include <iostream>

namespace forgecc
{

    enum class DiagLevel
    {
        Note,
        Warning,
        Error
    };

    class DiagnosticEngine
    {
    public:
        void report(DiagLevel level, const std::string &msg, int line = 0, int col = 0)
        {
            const char *tag = (level == DiagLevel::Error)     ? "error"
                              : (level == DiagLevel::Warning) ? "warning"
                                                              : "note";
            if (line > 0)
                std::cerr << "forgecc:" << line << ":" << col << ": " << tag << ": " << msg << "\n";
            else
                std::cerr << "forgecc: " << tag << ": " << msg << "\n";
        }
    };

} // namespace forgecc