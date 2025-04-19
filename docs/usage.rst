Usage
-----
For this example, start by loading Mauna Loa Weekly Atmospheric CO2 Data from `statsmodels <https://www.statsmodels.org>`__.

.. code:: python

   >>> from statsmodels.datasets import co2
   >>> data = co2.load().data
   >>> # Fill the few missing points in the dataset
   >>> data = data.ffill()

Use ``Autoperiod`` to find the list of periods based in this data (if any).

.. code:: python

   >>> from pyriodicity import Autoperiod
   >>> Autoperiod.detect(data)
   array([52])

The detected periodicity length is 52 which suggests a strong yearly seasonality given that the data
has a weekly frequency.

To confirm this, we run the ``Autoperiod`` again on the same data resampled to a monthly frequency.

.. code:: python

   >>> data = co2.load().data
   >>> data = data.resample("ME").mean().ffill()
   >>> Autoperiod.detect(data)
   array([12])

As expected, the detected periodicity length for the monthly frequency is 12, which confirms
the strong yearly seasonality of the data.

We can also use online detection methods for data streams as follows.

.. code:: python

   >>> data_stream = (sample for sample in data.values)
   >>> detector = OnlineACFPeriodicityDetector(window_size=128)
   >>> for sample in data_stream:
   ...   periods = detector.detect(sample)
   >>> 12 in periods
   True

As expected, the periodicity length of 12, corresponding to a yearly seasonality, was detected.

Note that online detection methods accept data samples of any length so that it is suitable for data stream
updates (i.e., single point updates), batch updates, or any arbitrary mixture of the two.

All the supported periodicity detection methods can be used in the same manner as in the examples
above with different optional parameters. Check the :doc:`api` for more details.
