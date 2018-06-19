from Cython.Distutils import build_ext
from distutils.core import setup, Extension

ext_modules=[
            Extension("ana_cont.src.pade",    # location of the resulting .so
                             ["ana_cont/src/pade.pyx"],) ]


setup(name='ana_cont',
      version='1.0',
      description='Analytic continuation package',
      author='Josef Kaufmann',
      url='https://github.com/josefkaufmann/ana_cont',
      packages=['ana_cont'],
      package_dir={'ana_cont':'ana_cont/'},
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      )

