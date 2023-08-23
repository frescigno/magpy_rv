[![Documentation Status](https://readthedocs.org/projects/MAGPy_RV/badge/?version=latest)](https://magpy-rv.readthedocs.io/en/latest/badge=latest)
[![PyPI version](https://badge.fury.io/py/magpy_rv.svg)](https://badge.fury.io/py/magpy_rv)
[![DOI](https://zenodo.org/badge/DOI/number.svg)](https://doi.org/number)

[![github](https://img.shields.io/badge/GitHub-frescigno-181717.svg?style=flat&logo=github)](https://github.com/frescigno)
[![website](https://img.shields.io/badge/Website-Federica_Rescigno-5087B2.svg?style=flat&logo=telegram)](https://frescigno.github.io)

# magpy_rv

Modeling Activity with Gaussian process regression in Python

Pipeline to model data with Gaussian Process regression and affine invariant Monte Carlo Markov Chain parameter searching algorith.
To use please cite the original publication (Rescigno et al. in review)


# Documentation

**Documentation Site:**  [MAGPy RV.readthedocs](https://magpy-rv.readthedocs.io/en/latest/)

# Installation

**Build conda environment**
MAGPy_RV can be run in its own environment. To generate it follow the steps:

Update dependencies in env.yml [file](env.yml)
Run the following from the folder containing the .yml file
``conda env create -f conda_env.yml``


**Package installation using pip**
Install pip (if Anaconda or miniconda is installe use ``conda install pip``) 

Install package   
``pip install magpy_rv``

# Examples
Examples are hosted [here](https://github.com/frescigno/magpy_rv/tree/main/docs/tutorials):  

1. [Simple GP Example](https://github.com/frescigno/magpy_rv/blob/main/docs/tutorials/(1)_no_model_tutorial.ipynb) shows the most basic code use.
   
2. [Polynomial Model](https://github.com/frescigno/magpy_rv/blob/main/docs/tutorials/(2)_polynomial_tutorial.ipynb) adds a model to the GP and introduces MCMC parameter search.
  
3. [Pegasi 51b](https://github.com/frescigno/magpy_rv/blob/main/docs/tutorials/(3)_51_peg_tutorial.ipynb) walks through the full rv analysis with a GP to model activity and Keplerians to model a planet.
    
4. [Offset](https://github.com/frescigno/magpy_rv/blob/main/docs/tutorials/(4)_offset_tutorial.ipynb):
full end-to-end pipeline to calculate 'sun-as-a-star' RVs and magnetic observables 