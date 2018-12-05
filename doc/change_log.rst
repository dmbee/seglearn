Change Log
==========

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