# Aura

## Development

> Note: If you're developing on VSCode, download the necessary extensions to make your life easier (e.g. Jupyter Notebooks, Rust Analyzer, PyLint)

Initiate your conda environment. To donwload Anaconda and manage your Python environments, go [here](https://www.anaconda.com/download). Here we're using Python 3.10

```bash
conda create -n aura python=3.10
conda activate aura // Run this every time you open your project
```

We'll be using `PyO3` + `Maturin` to integrate Rust and Python in the same package in case we need anything that executes fast. [(Read more here)](https://medium.com/@MatthieuL49/a-mixed-rust-python-project-24491e2af424)

Installing Python Dependencies

```bash
pip install -r requirements.txt
```

### Running Tests

Make sure you run this prior:
`maturin develop --extras tests`

<b>Testing Rust Source Code</b>

```bash
cargo test
```

If this doesn't work, try running:

```bash
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"
```

This is an issue with a PATH variable that Maturin is using to access a Python file. I'm trying to see if there's a way to make this automatic.
<br/>

<b>Running Python Tests</b>

```bash
conda install pytest # Only run this once
```

Running tests:

```bash
pytest
pytest aura/tests # (If the latter doesn't work)
```

### `/notebooks`

This is where all jupyter notebooks for testing models will be! AI folks work here mostly~ <3
