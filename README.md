# Course recommender


This study comprises a recommender system for postgraduate courses based on soft skills.

There are 5 Python files:


1. read_functions.py is a file that consists on multiple methods that are used to read and process the data for the recommendations.
2. combinatorics_brute_force.py is used to check each and every one of the combinations of the courses, and update a file every 2000 tested combinations.
2. script_test.py is the script for the various tests, which helped us empirically tune the hyperparameters of the Genetic Algorithms.
3. script_rec_function.py is the script which executes the genetic algorithms based on hyperparameters that need to be set either on console or through a bash script.
4. script_analysis.py is the script that reads and analyses the results in order to write the summary result files.


## Dependancies and configuration

This project requires the PyGAD framework, which needs to be installed.

To install PyGAD, simply use pip to download and install the library from PyPI (Python Package Index). The library is at PyPI at this page https://pypi.org/project/pygad.

Install PyGAD with the following command:

pip install pygad
To get started with PyGAD, please read the documentation at Read The Docs https://pygad.readthedocs.io.
(#editing-this-readme)!

Documentation soon™ (contact the authors if needed).