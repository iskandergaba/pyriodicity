.. raw:: html

   <div style="text-align: center;">

Pyriodicity
-----------

|PyPI Version| |PyPI - Python Version| |GitHub License| |CI Build| |Codecov|

Pyriodicity provides intuitive and easy-to-use Python implementation for periodicity detection in univariate signals.

.. raw:: html

   </div>

Supported Algorithms
~~~~~~~~~~~~~~~~~~~~

- `Autocorrelation Function (ACF) <https://otexts.com/fpp3/acf.html>`__
- `Autoperiod <https://doi.org/10.1137/1.9781611972757.40>`__
- `CFD-Autoperiod <https://doi.org/10.1007/978-3-030-39098-3_4>`__
- `Fast Fourier Transform (FFT) <https://otexts.com/fpp3/useful-predictors.html#fourier-series>`__

Installation
~~~~~~~~~~~~

To install the latest version of ``pyriodicity``, simply run:

.. code:: shell

   pip install pyriodicity

References
~~~~~~~~~~

-  [1] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting:
   principles and practice, 3rd edition, OTexts: Melbourne, Australia.
   `OTexts.com/fpp3 <https://otexts.com/fpp3>`__. Accessed on
   09-15-2024.
-  [2] Vlachos, M., Yu, P., & Castelli, V. (2005). On periodicity
   detection and Structural Periodic similarity. Proceedings of the 2005
   SIAM International Conference on Data Mining.
   `doi.org/10.1137/1.9781611972757.40 <https://doi.org/10.1137/1.9781611972757.40>`__.
-  [3] Puech, T., Boussard, M., D'Amato, A., & Millerand, G. (2020). A
   fully automated periodicity detection in time series. In Advanced
   Analytics and Learning on Temporal Data: 4th ECML PKDD Workshop,
   AALTD 2019, Würzburg, Germany, September 20, 2019, Revised Selected
   Papers 4 (pp. 43-54). Springer International Publishing.
   `doi.org/10.1007/978-3-030-39098-3_4 <https://doi.org/10.1007/978-3-030-39098-3_4>`__.

.. toctree::
   :hidden:
   :titlesonly:

   guide
   environment
   api

.. |PyPI Version| image:: https://img.shields.io/pypi/v/pyriodicity.svg?label=PyPI
   :target: https://pypi.org/project/pyriodicity/
.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/pyriodicity?label=Python
.. |GitHub License| image:: https://img.shields.io/github/license/iskandergaba/pyriodicity?label=License
.. |CI Build| image:: https://github.com/iskandergaba/pyriodicity/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/iskandergaba/pyriodicity/actions/workflows/ci.yml
.. |Codecov| image:: https://codecov.io/gh/iskandergaba/pyriodicity/graph/badge.svg?token=D5F3PKSOEK
   :target: https://codecov.io/gh/iskandergaba/pyriodicity
