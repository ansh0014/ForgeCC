from setuptools import setup, find_packages
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
readme_path = os.path.join(current_dir, "README.md")

long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="forgecc",
    version="0.2.3",
    description="High-Performance ML Compiler and Universal GPU Execution Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ForgeCC Team",
    author_email="team@forgecc.org",
    url="https://github.com/ansh0014/ForgeCC",
    project_urls={
        "Homepage": "https://github.com/ansh0014/ForgeCC",
        "Repository": "https://github.com/ansh0014/ForgeCC",
        "Documentation": "https://github.com/ansh0014/ForgeCC#readme",
    },
    license="MIT",
    keywords="compiler, machine-learning, deep-learning, cuda, gpu, tensor, pytorch, onnx",
    packages=find_packages(),
    package_data={
        "forgecc": ["*.dll", "*.so", "*.dylib"]
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
)
