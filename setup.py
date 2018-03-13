from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='seglearn',
      version='0.1',
      description='Machine Learning with Time Series Segmentation',
      author='David Burns',
      packages=find_packages(),
      include_package_data=True,
      install_requires=INSTALL_REQUIRES,
      author_email='david.mo.burns@gmail.com',
      url = 'https://github.com/dmbee/seglearn',
      license = 'BSD'
      )
