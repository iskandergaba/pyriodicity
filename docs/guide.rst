User Guide
----------

Start by loading a the ``co2`` emissions sample time series data from `statsmodels <https://www.statsmodels.org>`__.

.. code:: python

   >>> from statsmodels.datasets import co2
   >>> data = co2.load().data

You can then resample the data to whatever frequency you want. In this
example, we downsample the data to a monthly frequency

.. code:: python

   >>> data = data.resample("ME").mean().ffill()

Use ``Autoperiod`` to find the list of periods based in this data (if
any).

.. code:: python

   >>> from pyriodicity import Autoperiod
   >>> autoperiod = Autoperiod(data)
   >>> autoperiod.fit()
   array([12])

The detected periodicity length is 12 which suggests a strong yearly
seasonality given that the data has a monthly frequency.

You can also use ``CFDAutoperiod`` variant of ``Autoperiod`` or any
other supported periodicity detection method such as
``ACFPeriodicityDetector`` and ``FFTPeriodicityDetector`` and compare
results and performances.
