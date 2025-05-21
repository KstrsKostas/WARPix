# WARPix
*An interactive, JAX-powered black-hole shadow renderer (PyQt5 GUI).*

## Quick Start

cd WARPix
pip install -r assets/requirements.txt
python WARPix.py


## Features
* Scientifically accurate: Implements general relativistic ray-tracing for Kerr black holes.
* Multi-backend support: Runs efficiently on both GPU and CPU via JAX.
* Transparent output: Saves rendered shadows as transparent PNG images for easy integration.

## Examples

Here are two sample shadows WARPix can produce:

| Spin = 0.998, Angle = 90°                       | Spin = 0.829, Angle = 72°                       |
|:----------------------------------------------:|:----------------------------------------------:|
| ![Spin 0.998 θ90](https://github.com/user-attachments/assets/3604f1a2-2ac8-48f9-bd6e-4a2b30767b8b) | ![Spin 0.829 θ72](https://github.com/user-attachments/assets/c39b799d-1c43-44c9-bcb0-84c1c9266c91) |


