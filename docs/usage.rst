=====
Usage
=====

To use Private_swedish_mind in a project::

    import private_swedish_mind


After cloning code from the GitHub the codebase should look like:


.. code-block:: console

     ❯ tree -L 2
    ├── Untitled.ipynb
    ├── marina_tracks.ipynb
    └── private_swedish_mind            // package directory
        ├── AUTHORS.rst
        ├── CONTRIBUTING.rst
        ├── HISTORY.rst
        ├── LICENSE
        ├── MANIFEST.in
        ├── Makefile
        ├── README.rst
        ├── docs                        // documentation folder
        ├── hist.png
        ├── private_swedish_mind        // directory with package code
        ├── requirements.txt
        ├── requirements_dev.txt
        ├── setup.cfg
        ├── setup.py
        ├── tests                       // tests
        └── tox.ini


Next, create the Python virtual env:

.. code-block:: console

    $ python3 -m venv  private_mind


This will create the virtual environment for the  package. Activate it and check you are within the venv.

.. code-block:: console

    $ source private_mind/bin/activate
    $ which python


The output should point to the `XYZ/private_mind//bin/python`


Install packages needed:

.. code-block:: console

    pip3  install -r requirements.txt


