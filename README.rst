ana_cont: Python package for analytic continuation
=====

Update: The Wiki now contains descriptions and tutorials for the new graphical user interface.

Any questions, suggestions and bug reports will be received gratefully. 
(Mail to: josef *dot* kaufmann *at* tuwien *dot* ac *dot* at)

If you used this package to generate results for a publication, please cite 
`this paper by Geffroy, Kaufmann et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.127601>`_
(this
`bibtex file <https://github.com/josefkaufmann/ana_cont/wiki/bibtex/prl_dominique.bib>`_),


where some implementation details are described in the Supplementary Material. 

Short description
-----------------
Analytic continuation of Matsubara Green's functions is a difficult task,
for which there exists a large variety of methods and codes. 
This package does, for now, not contain any new method, but its concept
differs from most other codes: I do not provide compiled program that
is able to do many different tasks, but instead a small Python library
containing the necessary classes and functions. Based on this, the user
is encouraged to write their own scripts, fitting their needs.

Currently, the Pade approximation and the Maximum Entropy method
are implemented. 


Package structure
-----------------
* **ana_cont** contains the main code files.
* **gui** contains the code for the graphical user interface.
* **doc** contains some learning resources.
* **scripts** contains some simple examples how the library may be used; also the GUI executables are located there.

Requirements
-------------
The code was checked to run with the following versions:

* Python 3.7
* numpy 1.18.1
* scipy 1.4.1
* Cython 0.29.15 (only for Pade)
* matplotlib 3.1.3
* h5py 2.10.0
* PyQt5 (only for GUI)

It is likely that the code runs also with older package versions,
but I cannot guarantee for that.


Installation and usage
--------------

I recommend the following steps:

``git clone https://github.com/josefkaufmann/ana_cont``

If you want to use Pade, you have to compile:
``python setup.py build_ext --inplace``

Now you can add the code directory to the python path in your script
and import the package:

``sys.path.insert(0, '/path/to/ana_cont')``

``import ana_cont.continuation as cont``

The graphical user interface scripts can be executed as
``/path/to/ana_cont/scripts/maxent.py``
