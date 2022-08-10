Installation of ana_cont
========================

User installation
-----------------
Execute the following steps to install the dependencies
and the ``ana_cont`` library:

* ``sudo apt install python3-dev``
* ``python3 -m pip install -U pyqt5 matplotlib wheel cython h5py scipy``
* ``python3 -m pip install ana_cont``

This installs the ``ana_cont`` python library.
The GUI scripts are also installed and
can be called from the command line as ``maxent.py``, ``pade.py``, ``maxent_bosonic.py``.

Developer installation
----------------------
For a developer installation it is recommended to create a separate
virtual environment, based on python3. In this environment, perform the following steps:

* ``python3 -m pip install -U pip setuptools wheel numpy cython``
* ``git clone git@github.com:josefkaufmann/ana_cont.git``
* ``cd ana_cont``
* ``python3 -m pip install .``

Now you can run the unit tests:

``python3 -m unittest tests/ana_cont_tests.py -v``

