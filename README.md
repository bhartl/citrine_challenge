# Citrine-Challenge: constraint sampler

## Introduction
This is an implementation of the *Citrine - constraint sampler challenge* which is meant to provide a representative example of  math and engineering skills required to solve the problems in scope of the *Scientific Software Engineering* group at *Citrine*.

## Software

One of the core capabilities of *Citrine* is the ability to efficiently sample high dimensional spaces with complex, non-linear
constraints. In this challenge, applicants are asked to efficiently generate candidates that systematically explore as much of the valid
space as possible.

The “API” of the challenge is file based: the root directory of this project contains a `sample` *Python3* script which can be run as:
```shell script
./sampler <input_file> <output_file> <n_results>
```

Installation instructions can be found [below](#installation-instructions).

The sampler provides status messages about the sampling-state, i.e. 
it displays the current `sample id` (the current numer of sampled vectors),
  the `draw time [sec]` (the time it took to draw a single vector which satisfies the constraints), `total time [sec]` (the total sampling time) and a `status` message (*initialize*, *de-correlate*, *appending*, *finished* or a message with the reason of rejection: *atol* or *constraints*)

*Optional keyword arguments*:
- `--atol <float>` controls the minimum Eucledian distance (in normalized space) which all sample vectors must exhibit before being added to the samples list (positive float, defaults to 1e-4)
- `--n-correlated-samples <uint>`: number of samples considered in minimizing correlations (positive int, defaults to 100)
- `--normalize-pca-threshold <float>`: threshold of *PCA* variances which are considered as important in order to reduce the dimensionality of the constraint problem (semi positive float, defaults to 0.1)
- `--log-interval <int>` controls the interval how often the samling status is printed to the screen (positive integer, defaults to 10).

*Optional flags*:
- `-q` controls the verbosity of `./sampler`. Use `-q` to perform the sampling without screen output.
- `-u`: Use the example given by the `<input_file>` during sampling 

You can also ask for help:
```shell script
./sampler -h
```

### Input file
The input file starts with a single line header that gives the dimensionality of the problem, which
is defined on the unit hypercube. The next line is a single example feasible point. The remaining lines are a list of constraints as
python expressions containing `+` , `-` , `*` , `/` , and `**` operators.
They have been transformed such that they all take
the form `g(x) >= 0.0` .

```shell script
./sampler <input_file> <output_file> <n_results>
```

we can test the sampler on the [example](./test/files/example.txt) problem:
```shell script
./sampler test/files/example.txt test/dat/sampler/example.txt 1000
> sample 1000 configurations from input <test/files/example.txt>
>        sample id  draw time [sec] total time [sec]           status
>        1000/1000           0.0352          35.1512         finished
> 
> done with 1872 sampling steps after 35.15 seconds
> writing results to output file <test/dat/sampler/example.txt>
```

Other example files are listed here: [mixture](./test/files/mixture.txt), [example](./test/files/example.txt), [formulation](./test/files/formulation.txt) and [alloy](./test/files/alloy.txt).

### Output file
An output file is generated after the sampling finished. It contains a list of `<n_results>` vectors (space delimited within the vector; one vector per line). 
The challenge is that for evaluating for `n_results = 1000` the execution takes less than 5 minutes.

### Project structure
The project is based on the `citrine_challange` python module in the root directory, which contains the classes `Constraint`, `SamplerConstraint`, `BaseSampler` and `AdaptiveSampler`.
(Demonstrative) unittests can be found in `test/citrine_challenge` and example input files are located in `test/files`.
 
The `Constraint` was provided by *Citrine* which handles the parsing of input files.

The `SamplerConstraint` inherits the `Constraint` class and extends its functionality by giving access to the numerical values of constraints, `g(x)`, and its violations, `f(x) = -g(x) * h(-g(x))`, where `h(y)` is the Heaviside function satisfying `h(y) = 0` for `y<0` and `h(y) = 1` otherwise. I.e. `f(x) > 0` represents the violation of constraint `g(x)`, `f(x)=0` ensures that the constraint `g(x)>=0` is satisfied.
 
The `BaseSampler` class handles input, output and general sampling tasks and containts also a rather straight-forward random sampling method (which may be applicable for low-dimensional problems): 
  Random samples on the unit hypercube are drawn and evaluated against the constraints `g(x)>=0`. If they satisfy the constraints they are stored in a list, `{Xi}`, until `n_results` configurations have been identified, which is the current (most simple) convergence criterion. Note that we only append results to the sample list if their Eucledian distance to all other samples in this list is above a certain threshold `min(|{Xi} - x|) >= atol` (which is an optional parameter of the sampler, see `./sampler -h`).
  
The `AdaptiveSampler` class inherits `BaseSampler` and **represents the main sampler class of this project**: 
**First**, feasible parameter regions are explored using *scipy.optimize* to functionally minimize constraint-violations `f(x)` (provided by `SamplerConstraint`). 
Samples are drawn randomly and the minimization steers these configurations into the next local minimum with respect to constraint-violations, i.e. minimizing `f(x)` under variation of `x`(we use *scipy*'s *L-BFGS-B* minimization for that task).
This is done until a candidate sample, `c`, satisfies all constraints `f(c)=0`.
In order to explore broader regions of the configuration space a **second optimization procedure** tries to minimize [correlations]((https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html)) of the candidate sample `c` with a representative list `R{Xi}` of the sample list `{Xi}` (`corr(R{Xi}, c) -> min`), **always under consideration of the constraints `g(x)>=0`** (note that a different strategy could be to maximize the minimal Eucledian distance (`min|{Xi} - c| -> max`). 
Since the constraints of the problem may introduce a bias (offset and scales) of different components of the vectors in `{Xi}` the second minimization is evaluated in a **normalized space** of the sampled data `{Xi}`: 
First we use *scikit-learn*'s *StandardScaler* to remove the mean of `R{Xi}` and scale to unit variance: `{Xi} -> S({Xi})` .
On top of that we use Principal Component Analysis (*PCA*, again *scikit-learn*) for dimensional reduction by only considering the leading principal components of `S{Xi}` which's variances, `vi`, are larger than or equal to a specified threshold `vi >= normalize_pca_threshold`.
With this we have defined the transformation `N(x) = PCA(S(x))` and we define the constraint functional to optimize (i.e. the Lagrangian): `L(c) = |corr(R(N({Xi}), N(c))| + sum_i(f_i(c)) + sum_i(l_i * f_i(c))`, where the first sum over all violations `f_i(x)` supports the minimizer to not leave feasible regions and the second sum over lagrange multipliers `l_i` ensures that the constraints are satisfied `f_i(x)=0` (we use *scipy*'s *SLSQP* implementation to minimize `L(c)`, the first sum supports convergence since *SLSQP*'s constraint handling is not bulletproof).

*Remarks:*
The leading *PCA* threshold is controlled via the parameter`--normalize-pca-threshold` (float), note that we keep at least two dimensions).
For the moment `R(N{Xi})` is drawn at random from `N({Xi})`, the length is controlled by the parameter `--n-correlated-samples` (int). 
`AdaptiveSampler` uses an *initialization* phase in which `n_correlated_samples` are not optimized for correlation minimization (this is before normalization can effectively be  applied). These samples are subsequently overwritten in a *de-correlation* phase. After that newly drawn samples are merely *appended* until convergence is reached: until `n_results` samples have been identified (this could be done more sophisticated in the future).

The sampler is python module based, one can also use the `citrine_challenge` module in python scripts:
```python
from citrine_challenge import AdaptiveSampler as Sampler

sampler = Sampler(input_file="<input_file>", output_file="<output_file>", n_results="<n_results>")
results = sampler.sample()
sampler.dump_samples()
```
- if one promts to the `citrine_challenge` root directory 
- or adds it to the `PYTHONPATH` environment variable; 
- or one chooses to install the sampler using `python setup.py install` in the projects main directory)

### Installation instructions
The project is written in *Python 3* and was tested with *Python 3.7* and *Python 3.8.0* under *Ubuntu 16.04* and *Ubuntu 18.04*.

#### Get the project
Download the *git-repository* or clone it using
```shell script
git clone https://github.com/bhartl/citrine_challenge.git
```

#### Install Python using `Anaconda`
We recommend the use of *anaconda3*, which can be downloaded [here](https://www.anaconda.com/distribution/): download the *Python 3.7 version* of Anaconda and follow the install instructions.

In anaconda you can safely generate virtual environments (with minimum python packages) which do not interfere with your main python installation:

```shell script
conda create -n citrine-challenge
```
and then activate the virtual environment using
```shell script
source activate citrine-challenge
```
with 
```shell script
conda deactivate
```
you can go back to your orignal python environment.

#### Python module dependencies
The `citrine_challnge` implementation depends on the python modules `numpy`, `scipy`, `scikit-learn` (and optional `unittest2`) which can be installed using `conda`:
```shell script
conda install -c anaconda numpy scipy scikit-learn
```
and optional:
```shell script
conda install -c anaconda unittest2
```

If the use of anaconda is not desired, installation may be performed using `pip`
```shell script
python -m pip install --upgrade pip
pip install numpy scipy scikit-learn
```
(and possible dependecies thereof)
and optional
```shell script
pip install unittest2
```

#### Test installation
Tests are located in the `test/citrine_challenge` subfolder and represent demonstrative tests for the `BaseSampler` functionalities and apply `sample` and `dump_samples` methods for `BaseSampler` and `AdaptiveSampler` to the four provided test cases: [mixture](./test/files/mixture.txt), [example](./test/files/example.txt), [formulation](./test/files/formulation.txt) and [alloy](./test/files/alloy.txt). (Note that `BaseSampler` is incapable of sampling the *alloy* problem in a reasonable time.)

The installation may be tested using the `unittest2` package by executing:
```shell script
./test/sampler
```
or directly
```shell script
unit2 discover test/citrine_challenge/
```

*Remark*:
If you encounter warnings like *'RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility.'* you may update your `numpy == 1.16.1`, which, at the time of creation of this document, was not yet compatible with `python 3.8`.
I have encountered this issue for a while and I don't now a good solution, *numpy*'s suggestion is to ignore the warning. 

## Remarks
There might be a bias for certain problems with strongly attractive basins: Many solutions of the minimization step may end up in this region.

The execution time for the `test/files/alloy.txt` task is mainly due to the dimensionality in the de-correlation step. Including too many samples and too many components of the PCA transformed samples in the correlation matrix will slow down the process drastically.
This parameter is quite delicate.

## Licence
The software can be distributed under MIT [LICENCE](./LICENCE).
