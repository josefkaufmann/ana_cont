ana_cont
=================================
Package for Analytic Continuation of Green's Functions
======================================

#Purpose
The ana_cont package is a toolbox for analytic continuation of many-body Green's functions from the imaginary to the real axis. It is written in Python and can be used
in Python scripts.

There are already many excellent open-source codes for analytic continuation. The intention of this project is somewhat different:

* It is not a monolithic, immutable program; on the contrary, the user is required to use its classes and methods to compose their own scripts, suited to their needs. Thereby, complications and confusions about input and output formats are omitted entirely.
* It does not feature or promote any specific method for analytic continuation, but rather provides the possibility to compare results of different methods. Since analytic continuation is a difficult task, the author believes that one should sometimes compare results obtained with different methods.


#Main Classes
##AnalyticContinuationProblem
This is the class for defining analytic continuation problems. On user-level, only instances of this class are used. The variables held by the objects of this type are not specific to any continuation method or algorithm:

* Imaginary axis: Matsubara frequencies or bins of imaginary time, on which the input data is given.
* Real axis: Frequency values, on which the continued quantity should be evaluated.
* Imaginary-time/frequency data: The data that should be continued to the real axis.
* Kernel type: Specifies the analytic relation between imaginary- and real-axis data.


##AnalyticContinuationSolver
This is the base class, from which all the solvers inherit.

###MaxentSolverSVD
Currently, this is the main solver of the library.

###PadeSolver
