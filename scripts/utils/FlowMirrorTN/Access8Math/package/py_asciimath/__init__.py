import os
import platform

if (platform.system() == 'Windows'):
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__)).replace('\\', '/')
else:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

__version__ = "0.2.5"
