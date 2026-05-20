# storyteller
Andrej Karpathy's Storyteller

## Local setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Register the kernel for Jupyter:

```bash
python -m ipykernel install --user --name storyteller --display-name "Python (storyteller)"
```

Launch Jupyter:

```bash
jupyter lab
```
