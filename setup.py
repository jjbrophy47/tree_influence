import os
from setuptools import setup
from setuptools import find_packages

import numpy as np
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules = ['tree_influence/explainers/parsers/_tree32.pyx',
               'tree_influence/explainers/parsers/_tree64.pyx']

libraries = []
if os.name == 'posix':
    libraries.append('m')

setup(name="tree-influence",
      version="0.0.2",
      description="Influence Estimation for Gradient-Boosted Decision Trees",
      author="Jonathan Brophy",
      author_email="jonathanbrophy47@gmail.com",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/jjbrophy47/tree_influence",
      packages=find_packages(),
      include_package_data=True,
      package_dir={"tree_influence": "tree_influence"},
      classifiers=["Programming Language :: Python :: 3.9",
                   "Programming Language :: Python :: 3.10",
                   "License :: OSI Approved :: Apache Software License",
                   "Operating System :: OS Independent"],
      python_requires='>=3.9',
      install_requires=["numpy>=1.21",
                        "scikit-learn>=0.24.2"
                        "torch>=1.9.0"],
      ext_modules=cythonize(ext_modules,
                            compiler_directives={'language_level': 3},
                            annotate=True),
      include_dirs=np.get_include(),
      zip_safe=False)
