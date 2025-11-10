# scikit-fem Hands-on (Binder)

Launch in your browser (no install):

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/c-abird/scikit-fem-tutorial/HEAD)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/c-abird/scikit-fem-tutorial/blob/HEAD/notebooks/01_poisson_1d.ipynb)
[![Open in Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/c-abird/scikit-fem-tutorial)

This repository contains minimal Jupyter notebooks for a hands-on introduction to finite elements with scikit-fem, designed to run in the browser without any local installation.

## What’s inside
- notebooks/01_poisson_1d.ipynb: Poisson equation on the unit interval with Dirichlet BCs, assembled and solved using scikit-fem (P1 elements).

## How to use (no install)
- Click the Binder badge above to launch a live Jupyter session with all dependencies preinstalled.
- If Binder is slow or rate-limited, try Colab (first cell installs packages) or Codespaces.

## Local run (optional)
If you prefer local execution:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
Then open notebooks/01_poisson_1d.ipynb.

## Notes
- Examples avoid external meshing binaries; we use meshzoo or built-ins.
- Binder cold starts can take a few minutes; consider pre-opening the link before the session.
- Contributions welcome (typos, small improvements, extra notebooks).
