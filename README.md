# rd53b_analysis
Analysis of RD53b front-end ASIC for the ATLAS HL-LHC pixel detector upgrade

Table of Contents
===
<!--ts-->
   * [Requirements](#requirements)
   * [Installing Python](#installing-python)

<!--te-->

## Requirements

Here is a list:
   * python (>=3.6) (see [Installing Python](#installing-python))

## Installing Python

If you do not have python, or you do not have the requisite version, you can checkout [danny_installs_python](https://github.com/dantrim/danny_installs_python):
```bash
$ git clone https://github.com/dantrim/danny_installs_python.git
$ source danny_installs_python/compiler_flags.sh # only if you need (and especially if you are on Mac OSX)
$ source danny_installs_python/install_python.sh
```

**When using `python`, and developing shared `python` code, it is highly recommended to use virtual environments and local (i.e. owned by the user, not the system) installations of python. This is no joke.**

The reasons for this are clearly laid out [here](https://realpython.com/intro-to-pyenv/). One of the best ways that I
have encountered for doing this is via [`pyenv`](https://realpython.com/intro-to-pyenv/). If you have this installed, you can do:
```bash
$ git clone https://github.com/dantrim/danny_installs_python.git
$ source danny_installs_python/compiler_flags.sh # only if you need (and especially if you are on Mac OSX)
$ source danny_installs_python/pyenv_install_python.sh
```

Follow the discussion at [danny_installs_python](https://github.com/dantrim/danny_installs_python#pre-requisites) to understand how to obtain `python`'s own dependencies.
