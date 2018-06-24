from Cython.Distutils import build_ext
from setuptools import setup
from distutils.core import setup, Extension
import numpy
from Cython.Build import cythonize

cythonize("ana_cont/pade.pyx")
setup(name = 'ana_cont',
      version = '0.5',
      description = 'Analytic continuation package',
      author = 'Josef Kaufmann',
      author_email = 'josefkaufma@gmail.com',
      url = 'https://github.com/josefkaufmann/ana_cont',
      packages = ['ana_cont'],
      package_dir = {'ana_cont':'ana_cont/'},
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension('pade', \
                        sources = ['ana_cont/pade.c'], \
                        libraries=['m'], \
                        include_dirs=[numpy.get_include()])],
      include_package_data=True,
      zip_safe=False,
      setup_requires=['Cython'],
      install_requires=['numpy', 'scipy'])

