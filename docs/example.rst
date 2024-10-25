Example
-------

Start by loading Mauna Loa Weekly Atmospheric CO2 Data from `statsmodels <https://www.statsmodels.org>`__
and downsampling its data to a monthly frequency.

.. code:: python

   >>> from statsmodels.datasets import co2
   >>> data = co2.load().data
   >>> data = data.resample("ME").mean().ffill()

Use ``Autoperiod`` to find the list of periods based in this data (if any).

.. code:: python

   >>> from pyriodicity import Autoperiod
   >>> autoperiod = Autoperiod(data)
   >>> autoperiod.fit()
   array([12])

The detected periodicity length is 12 which suggests a strong yearly
seasonality given that the data has a monthly frequency.

You can use other estimation algorithms like ``CFDAutoperiod`` variant of ``Autoperiod``,
``ACFPeriodicityDetector``, or ``FFTPeriodicityDetector`` and compare results and performance.
All estimation algorithms can be used in the same manner as in the example above with different
optional parameters. Check the :doc:`api` for more details.
