__AUTHOR__ = 'Bahram Jafrasteh'
# This source code is used to identiy proper environment

from sys import platform
import os
if platform == "linux" or platform == "linux2":
    cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    source_folder = os.path.join(cur_path, "resource")
    desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
else:
    cur_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    source_folder = os.path.join(cur_path, "resource")
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')