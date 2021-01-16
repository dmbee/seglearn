Change Log
==========

Version 1.2.2
* Fix Pype initialization - pwoller
* Deprecation fixes
* Changed name for predict_segmented_series to predict_unsegment in pipe
* Added predict_as_series convenience function in pipe

Version 1.2.1

* Implement predict_segmented_series for Pype, which resamples the prediction to the original time series sampling
* Updated feature rep example to show usage
* Introduced Segment class to replace SegmentX and SegmentXY, and as a base class for SegmentXYForecast to avoid code duplication
* Fix nn_training_curve example

Version 1.1.1

* Fix for imbalanced-learn resamplers by Matthias Gazzari

Version 1.1.0

* Support for imbalanced-learn resamplers by Matthias Gazzari

Version 1.0.9

* allows for vector targets, to support one-hot or multi-label encoding
* supports unsorted time series data
* supports time series data with duplicate time stamps (fixing scipy interp sorting)
* fixed numpy deprecation warning for test_base

Version 1.0.8

* added function for creating TS_Data from pandas
* minor bug fixes

Version 1.0.7

* step parameter defined for Segment transformers, which can be used to specify the sliding
window segmentation instead of overlap parameter (Matthias Gazzari)
* bug fix - variance feature calculation

Version 1.0.6

* ``order`` parameter defined for Segment transformers. Using 'C' ordering improves ANN estimator performance.
* ``__repr__`` defined for callable feature classes
* ``FunctionTransformer`` contributed by Matthias Gazzari
* __repr__ attributed added to ``TemporalKFold`` by Matthias Gazzari
* ``InterpLongToWide`` contributed by Philip Boyer

Version 1.0.5

* ``FeatureRepMix`` transformer contributed by Matthias Gazzari (qtux)
* Added verbosity control to ``FeatureRep`` transformer

Version 1.0.4:

* Fixed bug with ``Interp`` transformer and improved its unit testing
* Added preprocessing module with ``TargetRunLengthEncoder`` transform
* Further features functions contributed by Matthias Gazzari (qtux)

Version 1.0.3:

* Continuous integration fixes
* Documentation and unit testing improvements

Version 1.0.2:

* Added further features

Version 1.0.1:

* Improvements to documentation, error checking, and unit tests
* feature_functions min and max changed to minimum and maximum respectively

Version 1.0.0:

* Major API change integrating preprocessing and estimation into single pipeline (Pype)
* Note this is unfortunately not backwards compatible
* Added Interp transformer for resampling time series

Version 0.2.0:

* Added transformer for segmenting data set with padding and / or truncation
* Implemented random_state initialization for SegPipe
* Corrected documentation for FeatureRep

Version 0.1.6:

* First stable version