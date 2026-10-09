from setuptools import setup, find_packages
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
readme_path = os.path.join(current_dir, "README.md")
if not os.path.exists(readme_path):
    readme_path = os.path.join(current_dir, "..", "README.md")

long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="forgecc",
    version="0.2.0",
    description="High-Performance ML Compiler and Universal GPU Execution Engine",
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
