import setuptools
from setuptools import setup

setup(
    name='pyBayesPPR',
    version='1.0.0',
    description='Bayesian Projection Pursuit Regression',
    url='https://github.com/gqcollins/pyBayesPPR.git',
    author='Gavin Collins',
    author_email='',
    license='BSD-3',
    packages=setuptools.find_packages(),
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy']
)
