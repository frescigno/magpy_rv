.. _install:

Installation
============

Build conda environment
MAGPy-RV can be run in its own environment. To generate it follow the steps:

Update dependencies in env.yml file. Run the following from the folder containing the .yml file

.. codeblock::

    conda env create -f conda_env.yml

Package installation using pip
Install pip (if Anaconda or miniconda is installed use ``conda install pip``)

Install package

.. codeblock::

    pip install magpy-rv