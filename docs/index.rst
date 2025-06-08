.. raw:: html

   <div style="text-align: center;">

Pyriodicity
-----------

|PyPI - Python Version| |GitHub License| |Codecov| |Docs| |CI Build|

Pyriodicity provides an intuitive and efficient Python implementation of periodicity length detection methods in univariate signals. You can check the supported detection methods in the :doc:`api`.

.. raw:: html

   </div>


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

#. Hyndman, R.J., & Athanasopoulos, G. (2021). Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. `OTexts.com/fpp3 <https://otexts.com/fpp3>`__. Accessed on 09-15-2024.
#. Vlachos, M., Yu, P., & Castelli, V. (2005). On periodicity detection and Structural Periodic similarity. Proceedings of the 2005 SIAM International Conference on Data Mining. `doi.org/10.1137/1.9781611972757.40 <https://doi.org/10.1137/1.9781611972757.40>`__.
#. Puech, T., Boussard, M., D'Amato, A., & Millerand, G. (2020). A fully automated periodicity detection in time series. In Advanced Analytics and Learning on Temporal Data: 4th ECML PKDD Workshop, AALTD 2019, Würzburg, Germany, September 20, 2019, Revised Selected Papers 4 (pp. 43-54). Springer International Publishing. `doi.org/10.1007/978-3-030-39098-3_4 <https://doi.org/10.1007/978-3-030-39098-3_4>`__.
#. Toller, M., Santos, T., & Kern, R. (2019). SAZED: parameter-free domain-agnostic season length estimation in time series data. Data Mining and Knowledge Discovery, 33(6), 1775-1798. `doi.org/10.1007/s10618-019-00645-z <https://doi.org/10.1007/s10618-019-00645-z>`__.
#. Wen, Q., He, K., Sun, L., Zhang, Y., Ke, M., & Xu, H. (2021, June). RobustPeriod: Robust time-frequency mining for multiple periodicity detection. In Proceedings of the 2021 international conference on management of data (pp. 2328-2337). `doi.org/10.1145/3448016.3452779 <https://doi.org/10.1145/3448016.3452779>`__.

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
