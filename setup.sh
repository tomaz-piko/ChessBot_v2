#!/bin/sh
python3 -m pip install -e .
python3 python/setup.py build_ext --inplace
python3 python/initialize.py