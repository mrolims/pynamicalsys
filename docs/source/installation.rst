Installation
============

Install **pynamicalsys** from PyPI to use the latest released version. If you
want to try unreleased changes or contribute to the package, use the
:ref:`development installation <install-development>` below.

Requirements
------------

- **Python 3.10 or newer** for the version described by this documentation.
- **pip**, the Python package installer.
- **Git**, only if you want to install from the source repository.

Dependencies such as NumPy and Numba are installed automatically by pip.
Compatibility with a particular Python version also depends on those
dependencies.

Run the installation commands in a terminal. Throughout this guide, ``python``
means the Python interpreter you want to use:

- On macOS and Linux, use ``python3`` if you are not in an activated environment.
  For example, ``python -m pip install pynamicalsys`` becomes
  ``python3 -m pip install pynamicalsys``.
- On Windows, you can use ``py`` if that is how you launch Python 3.
- In an activated virtual or conda environment, use ``python`` to select that
  environment's interpreter.

The ``-m pip`` form runs pip through the selected interpreter. Plain ``pip``
also works when it points to the same Python installation.

Create an environment (optional)
-------------------------------

A virtual environment keeps this project's packages separate from those used
by other projects. Creating one is optional: you can
:ref:`install directly into your chosen Python installation <install-release>`.
If you already have an environment for your work, activate it and skip this
section.

On macOS or Linux, create and activate an environment with:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate

On Windows, create the environment with:

.. code-block:: powershell

   py -m venv .venv

Then activate it in PowerShell:

.. code-block:: powershell

   .\.venv\Scripts\Activate.ps1

Or, if you use Command Prompt:

.. code-block:: bat

   .venv\Scripts\activate.bat

Use a Python installation that meets the requirement above when creating the
environment. After activation, check it with ``python --version``. Activate
this environment again whenever you open a new terminal to work on the project.

See the `Python virtual environment guide <https://docs.python.org/3/library/venv.html>`_
for more details.

If you use Anaconda or Miniconda, you can use an activated conda environment
instead. On Windows, run the installation commands in the Anaconda Prompt
with your chosen environment active.

.. _install-release:

Install the released version
----------------------------

Using your chosen Python interpreter, run:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install pynamicalsys

To upgrade an existing installation to the latest release:

.. code-block:: bash

   python -m pip install --upgrade pynamicalsys

Use the `stable documentation <https://pynamicalsys.readthedocs.io/en/stable/>`_
with the latest release. The development documentation may describe features
that are not available in the version installed from PyPI.

Notebooks and plotting examples
-------------------------------

Matplotlib is installed with **pynamicalsys**. Some tutorial plots also use
Seaborn for color palettes; install it if you want to run those examples:

.. code-block:: bash

   python -m pip install seaborn

The ``notebook`` extra adds IPython support for rendering model equations,
such as ``system.info["equation"]``, as typeset mathematics. It does not install
a notebook application. For versions that provide this extra, use:

.. code-block:: bash

   python -m pip install "pynamicalsys[notebook]"

If your released version does not provide the extra, you can install IPython
explicitly with ``python -m pip install ipython``. To run notebooks in JupyterLab,
install and launch it from the same environment:

.. code-block:: bash

   python -m pip install jupyterlab
   jupyter lab

Select a notebook kernel that uses the environment where you installed
**pynamicalsys**. In the development version, IPython is optional: without it,
``system.info["equation"]`` returns LaTeX source as a string. The
``system.info["equation_readable"]`` entry provides a plain-text alternative.

.. _install-development:

Install the development version
-------------------------------

To use the current source code, you need Git as well as Python. Clone the repository, then install from its root using your chosen Python
interpreter:

.. code-block:: bash

   git clone https://github.com/mrolims/pynamicalsys.git
   cd pynamicalsys
   python -m pip install .

For equation rendering support, use ``python -m pip install ".[notebook]"``
in place of the last command. Use the
`development documentation <https://pynamicalsys.readthedocs.io/en/latest/>`_
when working with the development version.

If you plan to edit the package, use an editable installation instead:

.. code-block:: bash

   python -m pip install -e ".[test,notebook]"

This also installs the test and notebook extras. An editable installation
uses the source in your checkout, so Python source edits are available when
you restart Python or your notebook kernel. See the :doc:`contributing` guide
for how to contribute, and pip's
`local installation guide <https://pip.pypa.io/en/stable/topics/local-project-installs/>`_
for details about editable installations.

Verify the installation
-----------------------

Run this in Python or a notebook to check the installed version and generate
a short trajectory:

.. code-block:: python

   import pynamicalsys
   from pynamicalsys import DiscreteDynamicalSystem

   print(pynamicalsys.__version__)

   system = DiscreteDynamicalSystem(model="logistic map")
   trajectory = system.trajectory(0.2, 5, parameters=[3.8])
   print(trajectory.shape)

The trajectory shape should be ``(5,)``. The first numerical calculation may
take longer because Numba compiles the routine before running it. Once the
example works, continue to the :doc:`quickstart`.

Troubleshooting
---------------

**Python cannot find pynamicalsys**
   Check that you activated the intended environment, then run
   ``python -m pip show pynamicalsys`` in the terminal. If it is not installed
   there, repeat the installation command in that environment.

**The terminal import works, but a notebook import fails**
   Run the following in the notebook to identify its Python interpreter:

   .. code-block:: python

      import sys
      print(sys.executable)

   Compare it with ``python -c "import sys; print(sys.executable)"`` in your
   activated terminal. Select the matching kernel and restart it after
   installing or upgrading packages.

**pip reports an incompatible Python version or cannot install a dependency**
   Check ``python --version`` and read the error to identify the package that
   could not be installed. Upgrade pip with
   ``python -m pip install --upgrade pip``. If the dependency does not support
   your Python version, create an environment with a version it supports that
   also meets pynamicalsys's minimum requirement.

**PowerShell blocks the activation script**
   You can use Command Prompt and its activation command above. Alternatively,
   invoke the environment's interpreter directly, for example
   ``.\.venv\Scripts\python.exe -m pip install pynamicalsys``.

If the problem persists, see :doc:`contact`. Include your operating system,
Python and pynamicalsys versions, the command you ran, and the full error
message so the issue can be reproduced.
