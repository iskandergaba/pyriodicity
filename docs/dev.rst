Development
-----------

This project is built and published using `Poetry <https://python-poetry.org>`__. To setup a development environment for this project you can follow these steps:

1. Install `Poetry <https://python-poetry.org/docs/#installing-with-pipx>`__.
2. Navigate to the root folder and install dependencies in a virtual environment:

.. code:: shell

   poetry install

3. If everything worked properly, you should have an environment under
   the name ``pyriodicity-py3.*`` activated. You can verify this by
   running:

.. code:: shell

   poetry env list

4. You can run tests using the command:

.. code:: shell

   poetry run pytest

5. To export the detailed dependency list, consider running the
   following:

.. code:: shell

   # Add poetry-plugin-export plugin to poetry
   poetry self add poetry-plugin-export

   # Export the package dependencies to requirements.txt
   poetry export --output requirements.txt

   # If you wish to include testing dependencies as well, run the following command
   poetry export --with test --output requirements-dev.txt

   # The same as above if you wish to export documentation dependencies
   poetry export --with docs --output requirements-docs.txt
