from setuptools import setup, find_packages
import os

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="forgecc",
    version="0.1.2",
    description="LLVM & CUDA Accelerated AI / Tensor Compiler Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
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

