# ForgeCC — PyPI Package Publishing Guide

This guide details the end-to-end process for building, packaging, and publishing `forgecc` to the Python Package Index (PyPI / pypi.org) using PyPI API tokens and `twine`.

---

## 1. Prerequisites and Tool Installation

Ensure `build` and `twine` are installed in the Python environment:

```cmd
py -m pip install --upgrade pip build twine wheel setuptools
```

---

## 2. Generating a PyPI API Token

1. Log in to [https://pypi.org](https://pypi.org).
2. Navigate to **Account Settings** -> **API tokens**.
3. Click **Add API token**.
4. Set the token scope to **Entire account** (for a new project) or **Project: forgecc** (for subsequent updates).
5. Copy the generated API token string (starts with `pypi-`).

---

## 3. Configuring the PyPI API Token

You can configure your token using environment variables or a configuration file.

### Method A: Setting Environment Variables (Recommended for CI/CD and CLI)

#### Windows (PowerShell)
```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-YOUR_EXACT_PYPI_API_TOKEN_HERE"
```

#### Windows (Command Prompt)
```cmd
set TWINE_USERNAME=__token__
set TWINE_PASSWORD=pypi-YOUR_EXACT_PYPI_API_TOKEN_HERE
```

#### Linux / macOS (Bash)
```bash
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-YOUR_EXACT_PYPI_API_TOKEN_HERE"
```

### Method B: Using `~/.pypirc` Configuration File

Create or edit the `.pypirc` file in your user home directory (`C:\Users\<Username>\.pypirc` on Windows or `~/.pypirc` on Linux/macOS):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_EXACT_PYPI_API_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_API_TOKEN_HERE
```

---

## 4. Building Distribution Packages

Before building, ensure the native shared library (`libforgecc_rt.dll` or `.so`) is compiled and copied into `python/forgecc/`:

```cmd
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build --target forgecc_rt_shared
Copy-Item build/runtime/libforgecc_rt.dll python/forgecc/libforgecc_rt.dll -Force
```

Clean previous distribution artifacts and build new Source Distribution (`.tar.gz`) and Wheel (`.whl`):

#### Windows (PowerShell)
```powershell
Set-Location python
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue dist, build, *.egg-info
py -m build
Set-Location ..
```

#### Linux / macOS (Bash)
```bash
cd python
rm -rf dist build *.egg-info
python -m build
cd ..
```

The output files will be created in `python/dist/`:
- `forgecc-0.2.0.tar.gz` (Source Distribution)
- `forgecc-0.2.0-py3-none-any.whl` (Built Wheel)

---

## 5. Validating Package Artifacts

Check package metadata and rendering integrity using `twine check`:

```cmd
py -m twine check python/dist/*
```

Expected output:
```text
Checking python/dist/forgecc-0.2.0-py3-none-any.whl: PASSED
Checking python/dist/forgecc-0.2.0.tar.gz: PASSED
```

---

## 6. Uploading to TestPyPI (Optional Sandbox Validation)

To verify the release on the staging repository before public deployment:

```cmd
py -m twine upload --repository testpypi python/dist/*
```

Install from TestPyPI to verify:

```cmd
py -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple forgecc
```

---

## 7. Uploading to Official PyPI (pypi.org)

Upload the package distribution to production PyPI:

```cmd
py -m twine upload python/dist/*
```

If `TWINE_USERNAME` and `TWINE_PASSWORD` are set, `twine` will authenticate automatically without prompting.

---

## 8. Verifying Live Installation

After publishing, verify the live package from PyPI:

```cmd
py -m pip install --upgrade forgecc
```

Verify in Python:

```cmd
py -c "import forgecc; print('ForgeCC Version:', forgecc.__version__); print('GPU Available:', forgecc.is_gpu_available())"
```
