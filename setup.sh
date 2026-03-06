#!/bin/bash
# Add src/ to PYTHONPATH so autograder can find ann/ and utils/
export PYTHONPATH="/autograder/source/src:$PYTHONPATH"
pip install -r requirements.txt