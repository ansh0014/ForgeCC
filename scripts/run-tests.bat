@echo off
echo ========================================
echo  ForgeCC - Test Runner
echo ========================================

echo.
echo [1] ForgeCC IR output (--emit-forge-ir)
echo ----------------------------------------
build\tools\forgecc\forgecc.exe test.cpp --emit-forge-ir

echo.
echo [2] LLVM IR output (--emit-ir)
echo ----------------------------------------
build\tools\forgecc\forgecc.exe test.cpp --emit-ir

echo.
echo [3] Normal compile mode
echo ----------------------------------------
build\tools\forgecc\forgecc.exe test.cpp -o output

echo.
echo ========================================
echo  All tests done
echo ========================================

