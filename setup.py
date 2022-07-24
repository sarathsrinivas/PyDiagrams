from setuptools import setup

setup(name='PyDiagrams',
      version='0.1',
      description='Feynman Diagrams computation for Quantum Many Body problem implemented in Python',
      url='http://github.com/sarathsrinivas/PyDiagrams.git',
      author='Sarath Srinivas S',
      author_email='srinix@pm.me',
      license='MIT',
      packages=['PyDiagrams'],
      install_requires=['torch', 'numpy',
      'PyQuadrature @ http://github.com/sarathsrinivas/PyQuadrature/tarball/main#egg=PyQuadrature',
      'PyGPR @ http://github.com/sarathsrinivas/PyGPR/tarball/main#egg=PyGPR' ],
      zip_safe=False)
