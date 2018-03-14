User Guide
==========

Introduction
------------

The seglearn python package is an extension to scikit-learn for multivariate sequential (or time series) data.

This package implements a sliding window segmentation, to create fixed length segments from the time series data. The segments can be learned directly with a neural network, or can  learned from a feature representation with any classical supervised learning algorithm in scikit learn. This package also supports learning datasets that include a combination of time series (sequential) data, and contextual data that is time independent with respect to a given time series.

This package provides an integrated pipeline from segmentation through to a final estimator that can be evaluated and optimized within the scikit learn framework.

Learning multivariate sequential data with the sliding window method is useful in a number of applications, including human activity recognition, electrical power systems, voice recognition, music, and many others. In general, this method is useful when the machine learning problem is not dependent on time series data remote from the window. However, the use of contextual variables or synthetic temporal variables (eg a moving average) can mitigate this limitation.

Time Series Data
----------------

This package supports multivariate sequential data, which in the case of time series data requires a regular sampling interval. Some time series datasets have irregular sampling and need time labels for each instance (eg purchase history on amazon.com). The methods available in this package are not suited to learning irregularly sampled time series, unless those time series would be amenable to re-sampling at regular intervals.

Sequential classification problems typically have a target variable associated with each sequence in the dataset (eg classification of song music genre). Similarly, contextual information may be associated with each time series (eg Artist) that doesn't change for the duration of the sequence and may be known to the classifier. Some classification problems may have a target that varies within a sequence (eg classification of which musical instruments are currently being played) - in this case, the sequence can be divided up into sub-sequences that have a constant target.

Regression datasets can similarly have a single target variable for each sequence (eg estimation of a song's sales on itunes).

Some classification and regression datasets will have a target that is itself a time series - as it varies continuously within the sequence. In these cases, both the time series variables and target variable need to be segmented. A decision needs to be made about where the target variable y should be sampled from for each instance in the segmented data set. Perhaps the target should be taken to be the middle value from the segment, the average over the segment, or maybe the value at the end of the segment. Maybe the target should be a sequence itself.

This package supports time series data sets with or without contextual information, and supports a target variable that is either fixed for each time series in the data set, or is itself a time series.

A final class of time series problems is forecasting. In which case, the goal may be to predict a future value or values of the target at some time remote to a given segment. This class of problems has continuous (time series) targets. In the sliding window segmentation approach, this problem can be conceptualized as predicting future segments (or segment values) from the current segment. Again contextual variables, or averaging variables can be used to bring information about segments prior to current segment into the forecast.


Why this Package
----------------

The algorithm to perform time series or sequence sliding window segmentation is straightforward. Without this package, segmentation can be performed outside of the scikit learn framework, and the machine learning algorithms applied to the segments directly or a feature representation of them with scikit-learn. However, algorithm performance is highly dependent on the sliding window segmentation parameters (window length, and overlap). This package integrates the segmentation into the pipeline so that parameter selection can be performed on the segmentation parameters, and the rest of the pipeline simultaneously.

The reason a new package is required to do this, instead of just a new transformer, is that time series segmentation changes the number of samples (instances) in the dataset. The final estimator sees one instance for each segment - which involves changing the number of samples and the target mid pipeline.

Changing the number of samples and the target vector mid-pipeline is not supported in scikit-learn - hence why this package is needed.


What this Package Includes
--------------------------

The main contributions of this package are:

1) ``SegmentX`` - transformer class for performing the time series / sequence segmentation when the target is contextual
2) ``SegmentXY`` - transformer class for performing the time series / sequence segmentation when the target is a time series. also handles forecasting.
3) ``FeatureRep`` - transformer class for computing a feature representation from segment data, and
4) ``SegPipe`` - pipeline class for integrating this for use with scikit learn machine learning algorithms and evaluation tools
5) ``TS_Data`` - an indexable / iterable class for storing time series & contextual data
6) ``split`` - a module for splitting time series or sequences along the temporal axis

What this Package Doesn't Include
---------------------------------

For now, this package does not include tools to help label time series data - which is a separate challenge.


Valid Sequence Data Representations
-----------------------------------

Time series data can be represented as a list or array of arrays as follows::

    >>> from numpy.random import rand
    >>> from np import array

    >>> # multivariate time series data: (N = 3, variables = 5)
    >>> X = [rand(100,5), rand(200,5), rand(50,5)]
    >>> # or equivalently as a numpy array
    >>> X = array([rand(100,5), rand(200,5), rand(50,5)])

The target, as a contextual variable (again N = 3) is represented as an array or list::

    >>> y = [2,1,3]
    >>> # or
    >>> y = array([2,1,3])


The target, as a continous variable (again N = 3), will have the same shape as the time series data::

    >>> y = [rand(100), rand(200), rand(50)]

The ``TS_Data`` class is provided as an indexable / iterable that can store time series & contextual data::

    >>> from seglearn.base import TS_Data
    >>> Xt = array([rand(100,5), rand(200,5), rand(50,5)])
    >>> # create 2 context variables
    >>> Xc = rand(3,2)
    >>> X = TS_Data(Xt, Xc)

There is a caveat for datasets that are a single time series. For compatibility with the seglearn segmenter classes, they need to be represented as a list::

    >>> X = [rand(1000,10)]
    >>> y = [rand(1000)]

If you want to split a single time series for train / test or cross validation - make sure to use one of the temporal splitting tools in ``split``. If you have many time series` in the dataset, you can use the sklearn splitters to split the data by series. This is demonstrated in the examples.


Using Seglearn
--------------

The package is relatively straightforward to use.

First see the `Examples <auto_examples/index.html>`_

If more details are needed, have a look at the `API Documentation <api.html>`_.


