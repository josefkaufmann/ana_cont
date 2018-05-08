from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
 
precomp_module = Extension('precomp', sources = ['ana_cont/src/precomp.c'], libraries=['m'],include_dirs=[numpy.get_include()])#{, extra_compile_args=['-std=c99', '-w'])
setup(name = 'precomp', version = '1.0', description = 'Precomputation of Maxent matrices.', ext_modules = [precomp_module])

setup(ext_modules = cythonize("ana_cont/src/pade.pyx"))
