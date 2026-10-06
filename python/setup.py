from setuptools import setup, find_packages

setup(
    name="forgecc",
    version="0.1.0",
    description="LLVM & CUDA Accelerated AI / Tensor Compiler Engine",
    author="ForgeCC Team",
    packages=find_packages(),
    package_data={
        "forgecc": ["*.dll", "*.so", "*.dylib"]
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

