# Exploring Exploration in Bayesian Optimization

This repository contains the official implementation of the paper:

**Exploring Exploration in Bayesian Optimization**  
*Leonard Papenmeier\*, Nuojin Cheng\*, Stephen Becker, Luigi Nardi*  
(*Equal contribution*)

## 📁 Repository Structure

```bash
exploring_exploration/
├── cython_extensions/
│   ├── tsp.pyx              # OTSD implementation (Cython)
│   └── oe.pyx               # OE implementation (Cython)
├── __init__.py
demo.py                      # Example script to compute and plot OTSD/OE
pyproject.toml               # Project metadata and dependencies
README.md                    # This file
```

## 📦 Installation
### 1. Install dependencies
We recommend using Poetry with Python 3.11+:
```bash
poetry install
```

If you prefer manual installation:
```bash
pip install numpy cython scipy matplotlib
```
Then build the Cython extensions:
```bash
python build.py
```

### 2. Run the demo
```bash
python demo.py
```
This script will:

* Generate random 2D points
* Compute the OTSD and OE metrics
* Plot their evolution across iterations

## 📚 Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{papenmeier2025exploring,
    title={Exploring Exploration in Bayesian Optimization},
    author={Papenmeier, Leonard and Cheng, Nuojin and Becker, Stephen and Nardi, Luigi},
    journal={arXiv preprint arXiv:2502.08208},
    year={2025}
}
```