Development
-----------

This project is built and published using `Poetry <https://python-poetry.org>`__. To setup a development environment for this project you can follow these steps:

1. Install `Poetry <https://python-poetry.org/docs/#installing-with-pipx>`__.
2. Navigate to the root folder and setup the virtual environment:

.. code:: shell

   poetry install

3. You should have the development environment activated now. Verify that you have an environment name starting with ``pyriodicity-py3`` by running:

.. code:: shell

   poetry env list

4. Run the unit tests using the command:

.. code:: shell

   poetry run pytest

5. To export the detailed dependency list, run the following:

.. code:: shell

   # Add poetry-plugin-export plugin to poetry
   poetry self add poetry-plugin-export

   # Export the package dependencies to requirements.txt
   poetry export --output requirements.txt

   # If you wish to include testing dependencies as well, run the following command
   poetry export --with test --output requirements-dev.txt

   # The same as above if you wish to export documentation dependencies
   poetry export --with docs --output requirements-docs.txt
