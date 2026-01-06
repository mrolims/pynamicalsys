Installation
============

This section guides you through installing the **pynamicalsys** package.

Prerequisites
-------------

- Python 3.8 or higher
- pip (Python package installer)

Install via PyPI
----------------

To install the latest stable release, run:

.. code-block:: bash

    $ pip install pynamicalsys

.. note::
    On **Windows**, it is **strongly recommended** to use `Anaconda <https://www.anaconda.com>`_. It simplifies dependency management and avoids potential issues with scientific libraries during installation.

    Be sure to run the command from the **Anaconda Prompt**, not from Command Prompt or PowerShell, to ensure the correct environment is activated.

Upgrade via PyPI
----------------

To upgrade your current version of **pynamicalsys** to the latest stable release, run in your command line:

.. code-block:: bash
    
    $ pip install pynamicalsys --upgrade

Install from source
-------------------

If you want to install the development version from the source repository, clone the repo and install with:

.. code-block:: bash

    $ git clone https://github.com/mrolims/pynamicalsys.git
    $ cd pynamicalsys
    $ pip install .

Verifying the installation
--------------------------

After installation, you can verify it by running Python and importing the package:

.. code-block:: python

    import pynamicalsys
    print(pynamicalsys.__version__)

Troubleshooting
---------------

If you encounter any issues, make sure you have the latest version of pip:

.. code-block:: bash

    $ pip install --upgrade pip build

For more help, visit the :doc:`contact` page.
