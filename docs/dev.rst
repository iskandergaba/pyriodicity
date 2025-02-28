Development
-----------

This project is built and published using `Poetry <https://python-poetry.org>`__. To setup a development environment for this project you can follow these steps:

1. Install `Poetry <https://python-poetry.org/docs/#installing-with-the-official-installer>`__.
2. Navigate to the root folder and install the project's virtual environment:

.. code:: shell

   poetry install

3. You should have the development environment activated now. Verify that you have an environment name starting with ``pyriodicity-py3`` by running:

.. code:: shell

   poetry env list

4. Format the code by running the command:

.. code:: shell

   poetry run ruff format

5. Check the code linting by running the command:

.. code:: shell

   poetry run ruff check

6. Run the unit tests by running the command:

.. code:: shell

   poetry run pytest

7. To export the detailed dependency list, run the following:

.. code:: shell

   # Add poetry-plugin-export plugin to poetry
   poetry self add poetry-plugin-export

   # Export the package dependencies to requirements.txt
   poetry export --output requirements.txt

   # If you wish to include development dependencies as well, run the following command
   poetry export --with dev --output requirements-dev.txt

   # The same as above if you wish to export documentation dependencies
   poetry export --with docs --output requirements-docs.txt
