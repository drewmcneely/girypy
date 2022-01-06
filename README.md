# Girypy: A library for creating Markov categories

This code is based heavily on the works of [Cho and Jacobs](https://arxiv.org/abs/1709.00322) and [Fritz](https://arxiv.org/abs/1908.07021).
It essentially provides an interface through abstract base classes to specify axioms for creating data types to represent Markov categories.
We provide a set of infix operators for composing, bimapping, and conditioning morphisms abstractly, among others.
This creates a syntax hopefully reminiscent of Notation 2.8 in Fritz.

## Installation

This code is not yet fully set up as a standalone package in PyPI.
However, I have set up a simple `setup.py` that should allow you to install in a `venv`.
