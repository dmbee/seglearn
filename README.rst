.. -*- mode: rst -*-

.. _scikit-learn: http://scikit-learn.org/stable/

.. _scikit-learn-contrib: https://github.com/scikit-learn-contrib

|Travis|_ |Pypi|_ |PythonVersion|_ |Coveralls|_ |Downloads|_

.. |Travis| image:: https://travis-ci.com/dmbee/seglearn.svg?branch=master
.. _Travis: https://app.travis-ci.com/github/dmbee/seglearn

.. |Pypi| image:: https://badge.fury.io/py/seglearn.svg
.. _Pypi: https://badge.fury.io/py/seglearn

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/seglearn.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/seglearn.svg

.. |Coveralls| image:: https://coveralls.io/repos/github/dmbee/seglearn/badge.svg?branch=master&&service=github
.. _Coveralls: https://coveralls.io/github/dmbee/seglearn?branch=master&service=github

.. |Downloads| image:: https://pepy.tech/badge/seglearn
.. _Downloads: https://pepy.tech/project/seglearn

seglearn
========

Seglearn is a python package for machine learning time series or sequences. It provides an integrated pipeline for segmentation, feature extraction, feature processing, and final estimator. Seglearn provides a flexible approach to multivariate time series and related contextual (meta) data for classification, regression, and forecasting problems. Support and examples are provided for learning time series with classical machine learning and deep learning models. It is compatible with scikit-learn_.

Documentation
-------------

Installation documentation, API documentation, and examples can be found on the
documentation_.

.. _documentation: https://dmbee.github.io/seglearn/

Dependencies
~~~~~~~~~~~~

seglearn is tested to work under Python 3.5, 3.6, and 3.8.
The dependency requirements are:

* scipy(>=0.17.0)
* numpy(>=1.11.0)
* scikit-learn(>=0.21.3)

seglearn is now also compatible with sklearn 1.0+

To run the examples, you need:

* matplotlib(>=2.0.0)
* keras (>=2.1.4) for the neural network examples
* pandas

In order to run the test cases, you need:

* pytest

The neural network examples were tested on keras using the tensorflow-gpu backend, which is recommended.

Installation
~~~~~~~~~~~~

seglearn-learn is currently available on the PyPi's repository and you can
install it via `pip`::

  pip install -U seglearn

or if you use python3::

  pip3 install -U seglearn

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies::

  git clone https://github.com/dmbee/seglearn.git
  cd seglearn
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/dmbee/seglearn.git

Testing
~~~~~~~

After installation, you can use `pytest` to run the test suite from seglearn's root directory::

  python -m pytest

Change Log
----------

Version history can be viewed in the `Change Log
<https://dmbee.github.io/seglearn/change_log.html>`_.

Development
-----------

The development of this scikit-learn-contrib is in line with the one
of the scikit-learn community. Therefore, you can refer to their
`Development Guide
<http://scikit-learn.org/stable/developers>`_.

Please submit new pull requests on the dev branch with unit tests and an example to
demonstrate any new functionality / api changes.

Citing seglearn
~~~~~~~~~~~~~~~

If you use seglearn in a scientific publication, we would appreciate
citations to the following paper::

  @article{arXiv:1803.08118,
  author  = {David Burns, Cari Whyne},
  title   = {Seglearn: A Python Package for Learning Sequences and Time Series},
  journal = {arXiv},
  year    = {2018},
  url     = {https://arxiv.org/abs/1803.08118}
  }


If you use the seglearn test data in a scientific publication, we would appreciate
citations to the following paper::

  @article{arXiv:1802.01489,
  author  = {David Burns, Nathan Leung, Michael Hardisty, Cari Whyne, Patrick Henry, Stewart McLachlin},
  title   = {Shoulder Physiotherapy Exercise Recognition: Machine Learning the Inertial Signals from a Smartwatch},
  journal = {arXiv},
  year    = {2018},
  url     = {https://arxiv.org/abs/1802.01489}
  }
