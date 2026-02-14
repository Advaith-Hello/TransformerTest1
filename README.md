# MNIST Vision Transformer in JAX

The project is a **Vision Transformer (ViT)** for classifying handwritten digits from the **MNIST** dataset created purely with **JAX**.

## Reasoning

The project was built in JAX to expose all the math that goes on behind the scenes while training a model like a ViT. In the project, all math is made using only jax functions, attention being manually implemented.

## Features

- Lightweight ViT architecture
- Built purely with JAX
- Many usable augmentations
- Extendable to other image-based datasets

## Requirements

- Python 3.9 to 3.12
- A proper jaxlib installation if using cuda

## Installation

```bash
# Clone the repo
git clone https://github.com/Advaith-Hello
cd TransformerTest1

# Setup venv and install dependencies

python -3.12 -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
