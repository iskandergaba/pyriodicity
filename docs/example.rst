Example
-------
For this example, start by loading Mauna Loa Weekly Atmospheric CO2 Data from `statsmodels <https://www.statsmodels.org>`__
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

To confirm this, we run the ``Autoperiod`` again on the same data resampled to a weekly and daily frequencies.

.. code:: python

   >>> data = co2.load().data
   >>> data = data.resample("W").mean().ffill()
   >>> autoperiod = Autoperiod(data)
   >>> autoperiod.fit()
   array([52])
   >>> data = co2.load().data
   >>> data = data.resample("D").mean().ffill()
   >>> autoperiod = Autoperiod(data)
   >>> autoperiod.fit()
   array([364])

As expected, the detected period lengths for weekly and daily frequencies are 52 and 364, respectively,
which confirms the strong yearly seasonality of the data.

You can use other estimation algorithms like ``CFDAutoperiod`` variant of ``Autoperiod``,
``ACFPeriodicityDetector``, or ``FFTPeriodicityDetector`` and compare results and performance.
All the supported estimation algorithms can be used in the same manner as in the example above
with different optional parameters. Check the :doc:`api` for more details.
