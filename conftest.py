# This file helps pytest and autograder find the src/ modules
import sys
import os

# Add src/ to path so ann/ and utils/ are importable
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)