from setuptools import find_packages, setup
import numpy
from Cython.Build import cythonize


with open("README.md", 'r') as f:
    long_description = f.read()


# python setup.py build_ext -i
setup(name='antco',
      version='0.1.2',
      license='BSD3',
      description='Ant Colony Optimization framework',
      author='Fernando García Gutiérrez',
      author_email='fegarc05@ucm.es',
      url=None,                                 # PENDING
      download_url='https://github.com/FernandoGaGu/Ant-Colony-Optimisation/archive/refs/tags/0.1.tar.gz',
      install_requires=[
                'numpy', 
                'joblib', 
                'cython', 
                'matplotlib', 
                'deap', 
                'seaborn', 
                'networkx'
            ],
      keywords=['Optimisation', 'Ant Colony Optimisation', 'Metaheuristic', 'Algorithms'],
      packages=find_packages(),
      ext_modules=cythonize(['**/*.pyx']),
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
