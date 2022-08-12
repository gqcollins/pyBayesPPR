import setuptools
from setuptools import setup

setup(name='pyBayesPPR',
      version='0.1',
      description='Bayesian Projection Pursuit Regression',
      url='http://www.github.com/gqcollins/BayesPPR',
      author='Gavin Collins',
      author_email='',
      license='BSD-3',
      packages=setuptools.find_packages(),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy'
      ]
      )
