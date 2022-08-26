#! /usr/bin/env python
"""A template for scikit-learn compatible packages."""

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('seglearn', '_version.py')
with open(ver_file) as f:
    exec(f.read())

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

DISTNAME = 'seglearn'
DESCRIPTION = 'A template for scikit-learn compatible packages.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'David Burns'
MAINTAINER_EMAIL = 'david.mo.burns@gmail.com'
URL = 'https://github.com/dmbee/seglearn'
LICENSE = 'BSD'
DOWNLOAD_URL = 'https://github.com/dmbee/seglearn'
VERSION = __version__
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: BSD License',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.8']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib',
        'keras',
        'pandas',
        'imbalanced-learn>=0.6.0',
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      include_package_data=True,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
