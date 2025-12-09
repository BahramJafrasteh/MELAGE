# -*- coding: utf-8 -*-
__author__='Bahram Jafraste'
from types import ModuleType
import sys
import os
# Get the current directory
current_directory = os.path.abspath(os.path.dirname(__file__))

# Add the current directory to the Python path
sys.path.append(current_directory)
from . import utils
from . import widgets
from . import widgets
from .melage import main
__version__ = '0.1.1'

