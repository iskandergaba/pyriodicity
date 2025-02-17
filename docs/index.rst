.. raw:: html

   <div style="text-align: center;">

Pyriodicity
-----------

|PyPI - Python Version| |GitHub License| |Codecov| |Docs| |CI Build|

Pyriodicity provides an intuitive and easy-to-use Python implementation for periodicity detection in univariate signals.

.. raw:: html

   </div>

Supported Algorithms
~~~~~~~~~~~~~~~~~~~~

- `Autocorrelation Function (ACF) <https://otexts.com/fpp3/acf.html>`__
- `Autoperiod <https://doi.org/10.1137/1.9781611972757.40>`__
- `CFD-Autoperiod <https://doi.org/10.1007/978-3-030-39098-3_4>`__
- `Fast Fourier Transform (FFT) <https://otexts.com/fpp3/useful-predictors.html#fourier-series>`__
- `RobustPeriod <https://doi.org/10.1145/3448016.3452779>`__

Installation
~~~~~~~~~~~~

To install ``pyriodicity``, simply run:

.. code:: shell

   pip install pyriodicity

To install the latest development version, you can run:

.. code:: shell

   pip install git+https://github.com/iskandergaba/pyriodicity.git

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
-  [4] Wen, Q., He, K., Sun, L., Zhang, Y., Ke, M., & Xu, H. (2021, June).
   RobustPeriod: Robust time-frequency mining for multiple periodicity detection.
   In Proceedings of the 2021 international conference on management of data (pp. 2328-2337).
   `https://doi.org/10.1145/3448016.3452779 <https://doi.org/10.1145/3448016.3452779>`__.

.. toctree::
   :hidden:
   :titlesonly:

   usage
   dev
   api

.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/pyriodicity?label=Python
.. |GitHub License| image:: https://img.shields.io/github/license/iskandergaba/pyriodicity?label=License
.. |Docs| image:: https://readthedocs.org/projects/pyriodicity/badge/?version=latest
   :target: https://codecov.io/gh/iskandergaba/pyriodicity
.. |Codecov| image:: https://codecov.io/gh/iskandergaba/pyriodicity/graph/badge.svg?token=D5F3PKSOEK
   :target: https://pyriodicity.readthedocs.io/en/latest
.. |CI Build| image:: https://github.com/iskandergaba/pyriodicity/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/iskandergaba/pyriodicity/actions/workflows/ci.yml
