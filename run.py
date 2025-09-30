#!/usr/bin/env python
"""
Wrapper script to run the logistics insights engine.
"""
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main

if __name__ == '__main__':
    main()
