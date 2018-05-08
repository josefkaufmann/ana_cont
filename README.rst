ana_cont: Package for analytic continuation of many-body Green's functions
=====

This is still a beta version, changes in the user interface cannot be excluded currently.
The main features have been tested extensively already and seem to work well. However, 
there might still be some minor issues. 

Any questions, suggestions and bug reports will be received gratefully. 
(Mail to: josef *dot* kaufmann *at* tuwien *dot* ac *dot* at)

Short description
-----------------
Analytic continuation of Matsubara Green's functions is a difficult task,
for which there exists a large variety of methods and codes. 
This package does, for now, not contain any new method, but its concept
differs from most other codes: I do not provide compiled program that
is able to do many different tasks, but instead a small Python library
containing the necessary classes and functions. Based on this, the user
is encouraged to write their own scripts, fitting their needs.

Currently, the Pade approximation and the Maximum Entropy method (classic and Bryan)
are implemented. 


Package structure
-----------------
* **ana_cont** contains the main code files.
* **doc** will contain a detailed description of the implemented formulas and code structure.
* **scripts** contains some simple examples how the library may be used. (More sophisticated 
   examples will be added later.) Also there are some scripts to generate test data.
* *install.sh* is a simple installation script that compiles the modules that are written in
   C and Cython, and it appends the directory to the PYTHONPATH environment variable. 
   
Requirements
-------------

* Python
* numpy
* scipy
* Cython (only for Pade)
