Gaussian-process Utility Thompson Sampling (GUTS)
=================================================

This repository contains the code implementing the GUTS algorithm.

Setup
=====

To correctly setup the repository, you need to first download the various
git submodules. To do this, you simply have to run the following command
in the project's main folder, after cloning the repository:

```
git submodule update --init --recursive
```

Requirements
============

Python **3** is required, to install all dependencies:

```
pip install -r requirements.txt
```

Experiments
===========

The main algorithm is located in `momabs/guts.py`. All parameters used for the bandit settings are listed there.

Simply running the `run.sh` script will run all the 20-arms experiments, with various cooldown, noise and number of objective settings.

Once the runs have finished, you can use the `plot.sh` script to generate the diverse plots that compare different settings against each other.

In order to make the 5-arms experiments, small changes need to be made to `guts.py`:

 - uncomment `mab.redef_self_5arm_example()` (line 161)
 - change the cooldown settings from 0,5,10,20 to 0,1,2,3 (line 223)
