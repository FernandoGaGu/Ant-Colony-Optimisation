# Ant Colony Optimisation (antco)

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


> Python package with a variety of implementations of ant-based algorithms for the optimisation of problems that can be represented as grarphs.

## Description

The antco package provides a set of computational tools as well as algorithms already implemented in Cython/Python for graph-based optimisation problems using ant-colony optimisation algorithms.

This package has been designed with the idea of providing the user with the flexibility to tackle complex problems using ant-based metaheuristics, a strategy that has proved highly successful for solve a wide variety of complex problems.

For more information about the module and its components, consult the Wiki pages.

## Installation

Requirements:
- numpy~=1.18.1
- joblib~=1.0.0
- cython~=0.29.21
- matplotlib~=3.3.1
- deap~=1.3.1
- seaborn~=0.11.0
- networkx~=2.5 

These are the versions with which the library has been developed, others different from those presented may work properly.

The simplest way to install the library is via pip:

```bash
pip install antco
```

To use the lastest version of the code:

```bash
git clone https://github.com/FernandoGaGu/Ant-Colony-Optimisation.git
```
and after cloning the repository run the setup.py script to compile the code written in Cython using:

```bash
python setup.py build_ext -i
```

## Usage

The dynamics of this package is quite simple, it is only necessary to define the problem and select the hyperparameters with which to run the optimisation. For more information on how to use this library, see the example ([examples](https://github.com/FernandoGaGu/Ant-Colony-Optimisation/tree/main/examples/).) codes or consult the Wiki pages where the module's operation is explained. 

## Warranties

The code used, although it has been reviewed and tested with different test problems where it has been shown to lead to correct solutions, is still under development and it is possible to experience undetected bugs. If any errors appear, please report them to us via <a href="https://github.com/FernandoGaGu/Ant-Colony-Optimisation/issues"> issues </a> 🙃.   


[contributors-shield]: https://img.shields.io/github/contributors/FernandoGaGu/Ant-Colony-Optimisation.svg?style=flat-square
[contributors-url]: https://github.com/FernandoGaGu/Ant-Colony-Optimisation/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/FernandoGaGu/Ant-Colony-Optimisation.svg?style=flat-square
[forks-url]: https://github.com/FernandoGaGu/Ant-Colony-Optimisation/network/members
[stars-shield]: https://img.shields.io/github/stars/FernandoGaGu/Ant-Colony-Optimisation.svg?style=flat-square
[stars-url]: https://github.com/FernandoGaGu/Ant-Colony-Optimisation/stargazers
[issues-shield]: https://img.shields.io/github/issues/FernandoGaGu/Ant-Colony-Optimisation.svg?style=flat-square
[issues-url]: https://github.com/FernandoGaGu/Ant-Colony-Optimisation/issues
[license-shield]: https://img.shields.io/github/license/FernandoGaGu/Ant-Colony-Optimisation.svg?style=flat-square
[license-url]: https://github.com/FernandoGaGu/Ant-Colony-Optimisation/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/Ant-Colony-Optimisation/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/GarciaGu-Fernando
