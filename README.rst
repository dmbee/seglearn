.. -*- mode: rst -*-

.. _scikit-learn: http://scikit-learn.org/stable/

.. _scikit-learn-contrib: https://github.com/scikit-learn-contrib

|Travis|_ |Pypi|_ |PythonVersion|_ |CircleCI|_ |Coveralls|_

.. |Travis| image:: https://travis-ci.org/dmbee/seglearn.svg?branch=master
.. _Travis: https://travis-ci.org/dmbee/seglearn

.. |Pypi| image:: https://badge.fury.io/py/seglearn.svg
.. _Pypi: https://badge.fury.io/py/seglearn

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/seglearn.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/seglearn.svg

.. |CircleCI| image:: https://circleci.com/gh/dmbee/seglearn.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/dmbee/seglearn/tree/master

.. |Coveralls| image:: https://coveralls.io/repos/github/dmbee/seglearn/badge.svg?branch=master&&service=github
.. _Coveralls: https://coveralls.io/github/dmbee/seglearn?branch=master&service=github


seglearn
========

Seglearn is a python package for machine learning time series or sequences using a sliding window segmentation. It provides an integrated pipeline for segmentation, feature extraction, feature processing, and final estimator. Seglearn provides a flexible approach to multivariate time series and contextual data for classification, regression, and forecasting problems. It is compatible with scikit-learn_.

Documentation
-------------

Installation documentation, API documentation, and examples can be found on the
documentation_.

.. _documentation: https://dmbee.github.io/seglearn/

Installation
------------

Dependencies
~~~~~~~~~~~~

seglearn is tested to work under Python 2.7 and Python 3.5.
The dependency requirements are based on the last scikit-learn release:

* scipy(>=0.13.3)
* numpy(>=1.8.2)
* scikit-learn(>=0.19.0)
* nose (nose>=1.1.2)

Additionally, to run the examples, you need:

* matplotlib(>=2.0.0)
* keras (>=2.1.4) for the neural network examples
* pandas

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

After installation, you can use `nose` to run the test suite::

  nosetests seglearn/tests/test_*

Development
-----------

The development of this scikit-learn-contrib is in line with the one
of the scikit-learn community. Therefore, you can refer to their
`Development Guide
<http://scikit-learn.org/stable/developers>`_.

About
-----

This package was developed by::

    David M. Burns MD, PhD(c)
    Sunnybrook Research Institute
    University of Toronto
    Email: david.mo.burns@gmail.com


Citing seglearn
~~~~~~~~~~~~~~~

If you use seglearn in a scientific publication, we would appreciate
citations to the following paper::

  @article{arXiv:1802.01489
  author  = {David Burns, Nathan Leung, Michael Hardisty, Cari Whyne, Patrick Henry, Stewart McLachlin},
  title   = {Shoulder Physiotherapy Exercise Recognition: Machine Learning the Inertial Signals from a Smartwatch},
  journal = {arXiv},
  year    = {2018},
  url     = {https://arxiv.org/abs/1802.01489}
  }
