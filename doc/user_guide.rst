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

Some classification and regression datasets will have a target that varies continuously within the sequence. In some cases, the goal may be to predict a future value (time series forecasting) which is equivalent. Currently, I have only used this package on datasets where the target variable is fixed for each sequence. It should be relatively straightforward to extend the work to datasets with continuously varying target.


Why this Package
----------------

The algorithm to perform time series or sequence sliding window segmentation is straightforward. Without this package, segmentation can be performed outside of the scikit learn framework, and the machine learning algorithms applied to the segments directly or a feature representation of them with scikit-learn. However, algorithm performance is highly dependent on the sliding window segmentation parameters (window length, and overlap). This package integrates the segmentation into the pipeline so that parameter selection can be performed on the segmentation parameters, and the rest of the pipeline simultaneously.

The reason a new package is required to do this, instead of just a new transformer, is that time series segmentation changes the number of samples (instances) in the dataset. The final estimator sees one instance for each segment - which involves changing the number of samples and the target mid pipeline.

Changing the number of samples and the target vector mid-pipeline is not supported in scikit-learn - hence why this package is needed.


What this Package Includes
--------------------------

The main contributions of this package are:

1) ``Segment`` - class for performing the time series / sequence segmentation
2) ``FeatureRep`` - class for computing a feature representation from segment data, and
3) ``SegPipe`` - pipeline for integrating this for use with scikit learn machine learning algorithms and evaluation tools


What this Package Doesn't Include
---------------------------------

For now, this package does not include tools to help label time series data - which is a separate challenge.

Using Seglearn
--------------

The package is relatively straightforward to use.

First see the `Examples <auto_examples/index.html>`_

If more details are needed, have a look at the `API Documentation <api.html>`_.


