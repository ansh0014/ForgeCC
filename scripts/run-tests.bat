@echo off
echo ========================================
echo  ForgeCC - Test Runner
echo ========================================

echo.
echo [1] ForgeCC IR output (--emit-forge-ir)
echo ----------------------------------------
build\tools\forgecc\forgecc.exe test.cpp --emit-forge-ir

echo.
echo [2] LLVM IR output with real CPU calls (--emit-ir)
echo ----------------------------------------
build\tools\forgecc\forgecc.exe test.cpp --emit-ir

echo.
echo [3] Emit object file (--emit-obj)
echo ----------------------------------------
build\tools\forgecc\forgecc.exe test.cpp --emit-obj -o forgecc_out
if exist forgecc_out.o (
    echo Object file written: forgecc_out.o
) else (
    echo No object file found.
)

echo.
echo [4] Normal compile mode
echo ----------------------------------------
build\tools\forgecc\forgecc.exe test.cpp -o output

echo.
echo ========================================
echo  All tests done
echo ========================================
