## Overview
mlrose-ky is a Python package for applying some of the most common randomized optimization and search algorithms to a range of different optimization problems, over both discrete- and continuous-valued parameter spaces.
### Project Background
mlrose-ky is a fork of the `mlrose-hiive` repository, which itself was a fork of the original `mlrose` repository. The original mlrose package was developed to support students of Georgia Tech's OMSCS/OMSA offering of CS 7641: Machine Learning.

This repository includes implementations of all randomized optimization algorithms taught in the course, as well as functionality to apply these algorithms to integer-string optimization problems, such as N-Queens and the Knapsack problem; continuous-valued optimization problems, such as the neural network weight problem; and tour optimization problems, such as the Travelling Salesperson problem. It also has the flexibility to solve user-defined optimization problems.
### Main Features
**Randomized Optimization Algorithms**
* Implementations of: hill climbing, randomized hill climbing, simulated annealing, genetic algorithm and (discrete) MIMIC;
* Solve both maximization and minimization problems;
* Define the algorithm’s initial state or start from a random state;
* Define your own simulated annealing decay schedule or use one of three pre-defined, customizable decay schedules: geometric decay, arithmetic decay or exponential decay.

**Problem Types**
* Solve discrete-value (bit-string and integer-string), continuous-value and tour optimization (travelling salesperson) problems;
* Define your own fitness function for optimization or use a pre-defined function.
* Pre-defined fitness functions exist for solving the: One Max, Flip Flop, Four Peaks, Six Peaks, Continuous Peaks, Knapsack, Travelling Salesperson, N-Queens and Max-K Color optimization problems.

**Machine Learning Weight Optimization**
* Optimize the weights of neural networks, linear regression models and logistic regression models using randomized hill climbing, simulated annealing, the genetic algorithm or gradient descent;
* Supports classification and regression neural networks.

### Project Improvements and Updates
The `mlrose-ky` project is undergoing significant improvements to enhance code quality, documentation, and testing. Below is a list of tasks that have been completed or are in progress:

1. **Fix Python Warnings and Errors**: All Python warnings and errors have been addressed, except for a few unavoidable ones like "duplicate code." ✅
	
2. **Add Python 3.10 Type Hints**: Type hints are being added to all function and method definitions, as well as method properties (e.g., `self.foo: str = 'bar'`), to improve code clarity and maintainability.
    
3. **Enhance Documentation**: NumPy-style docstrings are being added to all functions and methods, with at least a one-line docstring at the top of every file summarizing its contents. This will make the codebase more understandable and easier to use for others.
    
4. **Increase Test Coverage**: Tests are being added using Pytest, with a goal of achieving 100% code coverage to ensure the robustness of the codebase.
    
5. **Resolve TODO/FIXME Comments**: A thorough search is being conducted for any TODO, FIXME, or similar comments, and their respective issues are being resolved.
    
6. **Optimize Code**: Vanilla Python loops are being optimized where possible by vectorizing them with NumPy to enhance performance.
    
7. **Improve Code Quality**: Any other sub-optimal code, bugs, or code quality issues are being addressed to ensure a high standard of coding practices.
    
8. **Clean Up Codebase**: All commented-out code is being removed to keep the codebase clean and maintainable.

### Installation
mlrose-ky was written in Python 3 and requires NumPy, SciPy and Scikit-Learn (sklearn).

The latest released version is available at the [Python package index](#) and can be installed using pip:

``` pip
pip install mlrose-ky
```

Once it is installed, simply import it like so:
```python
import mlrose_ky
```

### Licensing, Authors, Acknowledgements
mlrose-ky was forked from the `mlrose-hiive` repository, which was a fork of the original `mlrose` repository.

The original `mlrose` was written by Genevieve Hayes and is distributed under the [3-Clause BSD license](https://github.com/gkhayes/mlrose/blob/master/LICENSE).

You can cite mlrose-ky in research publications and reports as follows:

- Nakamura, K. (2024). _**mlrose-ky: Machine Learning, Randomized Optimization, and SEarch package for Python**_. [https://github.com/knakamura13/mlrose-ky](https://github.com/knakamura13/mlrose-ky). Accessed: _day month year_.

Please also keep the original authors' citations:

- Rollings, A. (2020). _**mlrose: Machine Learning, Randomized Optimization and SEarch package for Python, hiive extended remix**_. [https://github.com/hiive/mlrose](https://github.com/hiive/mlrose). Accessed: _day month year_.
- Hayes, G. (2019). _**mlrose: Machine Learning, Randomized Optimization and SEarch package for Python**_. [https://github.com/gkhayes/mlrose](https://github.com/gkhayes/mlrose). Accessed: _day month year_.

Thanks to David S. Park for the MIMIC enhancements (from [https://github.com/parkds/mlrose](https://github.com/parkds/mlrose)).

BibTeX entry:
```default
@misc{Nakamura24,
 author = {Nakamura, K.},
 title  = {{mlrose-ky: Machine Learning, Randomized Optimization and SEarch package for Python}},
 year   = 2024,
 howpublished = {\url{https://github.com/knakamura13/mlrose-ky}},
 note   = {Accessed: day month year}
}

@misc{Hayes19,
 author = {Hayes, G},
 title  = {{mlrose: Machine Learning, Randomized Optimization and SEarch package for Python}},
 year   = 2019,
 howpublished = {\url{https://github.com/gkhayes/mlrose}},
 note   = {Accessed: day month year}
}
```
