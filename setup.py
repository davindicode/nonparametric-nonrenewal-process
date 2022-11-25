from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.0.1'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
#with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
#    all_reqs = f.read().split('\n')

#install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
#dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(name='neuroppl',
      author=['David Liu'],
      version=__version__,
      description='Probabilistic models for neural coding implemented in PyTorch',
      long_description=long_description,
      license='MIT',
      install_requires=[
          'numpy>=1.7',
          'torch>=1.4.0',
          'scipy>=1.0.0',
          #'pyro-ppl>=1.3.0',
          'sklearn'
          'tqdm>=4.36',
          'matplotlib',
          'imageio',
          'ipywidgets',
      ],
      python_requires='>=3.5',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
      ],
      packages=find_packages(),
)
