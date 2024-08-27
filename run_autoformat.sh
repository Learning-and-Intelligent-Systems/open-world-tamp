#!/bin/bash

# Exclude files in third_party folder for Black
python -m black . --exclude tamp/

# Exclude files in third_party folder for isort
isort . --skip tamp

# Exclude files in third_party folder for docformatter
docformatter -i -r . --exclude venv --exclude tamp
