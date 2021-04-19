from distutils.core import setup
from Cython.Build import cythonize
import numpy

# python setup.py build_ext -i
setup(name='antco',
      packages=['antco'],
      version='0.1',
      license='BSD3',
      description='Ant Colony Optimization framework',
      author='Fernando García Gutiérrez',
      author_email='fegarc05@ucm.es',
      url=None,                                 # PENDING
      download_url=None,                        # PENDING
      install_requires=[],                      # PENDING
      keywords=['Optimisation', 'Ant Colony Optimisation', 'Metaheuristic', 'Algorithms'],
      packages=['antco.c_utils'],
      ext_modules=cythonize('**/*.pyx'),
      include_dirs=[numpy.get_include()],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: BSD License', 
          'Programming Language :: Python :: 3', 
          'Programming Language :: Python :: 3.4', 
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',],
      )
