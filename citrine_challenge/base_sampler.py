import os
import numpy as np
from time import time
from citrine_challenge import SamplerConstraint


class BaseSampler(object):
	"""A rather straight forward attempt to tackle the Citrine sampling challenge by randomly choosing vectors
	from a n-dimensional hyper cube. The class is envisaged as **base class for other, more sophisticated samplers**.

	Once initialized from a proper `input_file` (given the `ouptut_file` and the number of requested samples, `n_results`,
	(along with other keyword-arguments helping with the sampler's configuration), it provides

	- the `BaseSampler.sample` method, which generates a list of `n_results` samples satisfying the constraints defined in the `input_file`;
	- the `BaseSampler.dump_samples` method, which writes the results to the specified `output_file`;
	- the `BaseSampler.main` class-method, which performs sampling on the `input_file`-constraint file and creates an `output_file`.

	The `BaseSampler.draw_sample` method (called by the `BaseSampler.sample` method) **is supposed to be overwritten**.
	"""

	def __init__(self, input_file: str, output_file: str, n_results: int, verbose=False, log_interval=None, max_draw_steps=10000, atol=1e-6):
		""" Construct a AdaptiveSampler object from a constraints file,

		:param input_file: Name of the file to read the Constraint from (string)
		:param output_file: Name of the output file (string)
		:param n_results: Number of output vectors (positive integer)
		:param verbose: Boolean which controls verbosity, i.e. if supporting information is printed during execution
		:param log_interval: Integer interval to display log messages. No log messages are printed if `log_interval` is None or 0, defaults to None
		:param max_draw_steps: Integer which controls the maximum number of attempts to draw new examples (in order to avoid endless loops)
		:param atol: Positive float value which controls the minimum Eucledian distance which all sample vectors must exhibit before being added to the samples list
		"""

		self._input_file = None       # input file path, str
		self._constraints = None      # Constraints object, initialized by input_file property
		self.input_file = input_file
		
		self._output_file = None      # output file path, str
		self.output_file = output_file
		
		self._n_results = 0           # number of results, int
		self.n_results = n_results

		self._samples = []            # list of samples
		self._log_interval = log_interval
		self._log = self._log_interval not in (0, None)
		self._verbose = verbose

		self._max_draw_steps = max_draw_steps
		self._atol = None
		self.atol = atol

	@property
	def input_file(self) -> str:
		"""Get name of the file to read the Constraint from (string)"""
		return self._input_file

	@input_file.setter
	def input_file(self, input_file: str):
		"""Set name of the file to read the Constraint from (string),
		`citrine_challenge.SamplerConstraints` container is initialized

		:param input_file: Name of the input file (string)
		"""

		assert os.path.isfile(input_file), "couldn't find input_file <{}>.".format(input_file)

		self._input_file = input_file
		self._constraints = SamplerConstraint(self._input_file)

	@property
	def output_file(self) -> str:
		"""Get name of the output file (string)"""
		return self._output_file

	@output_file.setter
	def output_file(self, output_file: str):
		"""Set name of the output file (string)

		:param output_file: Name of the output file (string)
		"""

		assert isinstance(output_file, str), "`output_file` must be of type `str`."
		self._output_file = output_file

	@property
	def n_results(self) -> int:
		"""Get number of output vectors (positive integer)"""
		return self._n_results

	@n_results.setter
	def n_results(self, n_results: int):
		"""Set number of output vectors (positive integer)

		:param n_results: Number of output vectors (positive integer)
		"""

		assert isinstance(n_results, int), "`n_results` must be of type `int`."
		self._n_results = n_results

	@property
	def atol(self) -> float:
		""" Get float value which controls the minimum Eucledian distance
		which all sample vectors must exhibit before being added to the samples
		list."""

		return self._atol

	@atol.setter
	def atol(self, atol):
		""" Set positive float value which controls the minimum Eucledian distance
		which all sample vectors must exhibit before being added to the samples
		list."""

		assert atol >= 0
		self._atol = atol

	@property
	def constraints(self) -> SamplerConstraint:
		"""Get `SamplerConstraint` instance, initialized by `input_file` property"""
		return self._constraints

	@property
	def converged(self) -> bool:
		"""Convergence criterion for sampling method `self.sample()`:
		if this method has a positive result (True) the sampling loop stops and convergence is assumed.

		This method can be used, i.e. overwritten, in more sophisticated implementations (not done here):
		If `n_results` samples are drawn one could systematically replace existing samples which are
		close in configuration space by newly drawn samples.

		:return: `True` if len of samples is desired number of samples, `False` otherwise.
		"""
		return len(self._samples) >= self.n_results

	def log(self, *args, **kwargs):
		"""Logging method which could also be used for writing log-messages to files.
		If the Sampler instance was initialized with `verbose=True` this method operates
		like the standard print function.
		"""
		if self._verbose:
			print(*args, **kwargs)

	def draw_sample(self, seed=None, **kwargs) -> tuple:
		"""Most basic sampler: draw samples at random until constraints are satisfied (MEANT TO BE OVERWRITTEN)

		:param seed: seed for random number generator, defaults to None
		:param kwargs: possible kwargs for inheriting classes
		:return: (sample array, number of draw attempts)
		"""

		# initialize random number generator
		if seed is not None:
			np.random.seed(seed)

		sample = None
		attempts = 0
		done = False  # status variable if sample satisfies constraints

		# search until new candidate sample is found (or self._max_draw_steps is reached)
		while not done:

			# draw random sample on n_dim hypercube
			sample = np.random.rand(self.constraints.n_dim)
			attempts += 1

			# evaluate, if constraints are satisfied
			done = self.constraints.apply(sample)

			#  check if max_draw_steps are exceeded
			if attempts > self._max_draw_steps and not done:
				break

		return sample, attempts

	@property
	def samples(self) -> list:
		"""Current list of drawn sample vectors satisfying constraints."""
		return self._samples

	def sample(self, initial_samples=(), **draw_kwargs) -> list:
		"""Perform sampling on constraint problem, loops until `self.converged` is `True` and calls `self.draw_sample`
		at each iteration. If newly drawn samples satisfy the constraints given by `self.input_file` and
		`self.append_criteria`, they are appended to the current list of samples, `self.samples`.

		:param initial_samples: initial list of sample vectors, defaults to `()`
		:param draw_kwargs: keyword arguments forwarded to `self.draw_sample` method
		:return: list of drawn sample vectors satisfying the constraints given by `self.input_file`
		"""

		self.log('sample {} configurations from input <{}>'.format(self.n_results, self.input_file))

		if self._log:
			self.log('{:>16s} {:>16s} {:>16s} {:>16s}'.format('sample id', 'draw time [sec]', 'total time [sec]', 'status'))

		self._samples = []  # initialize list of sample vectors
		for initial_sample in  initial_samples:
			# make sure initial samples, if provided satisfy constraints
			self.append_sample(initial_sample)

		sample_steps = 0    # number of total sampling steps (until convergence_criteria are satisfied)
		total_time = 0.     # total time of sampling (for logging)

		# loop until converged
		while not self.converged:

			if self._log and not np.mod(sample_steps, self._log_interval):
				self.log('\r{:>16s} '.format("{}/{}".format(len(self._samples) + 1, self.n_results)), end='')

			# draw n_dim random numbers on unit-hypercube
			start_sampling = time()
			sample, attempts = self.draw_sample(**draw_kwargs)

			# try to append sample
			status = self.append_sample(sample)

			# take time for logging (append_sample can also be time-consuming if a complex measure is evaluate)
			done_sampling = time()

			draw_time = done_sampling - start_sampling
			total_time += draw_time

			if self._log and not np.mod(sample_steps, self._log_interval):
				self.log('{:16.4f} {:16.4f} {:>16s}'.format(draw_time, total_time, status), end='')

			sample_steps += attempts

		# final log message
		if self._log:
			avg_time = total_time/len(self._samples)
			n_samples = "{}/{}".format(len(self._samples), self.n_results)
			self.log('\r{:>16s} {:16.4f} {:16.4f} {:>16s}'.format(n_samples, avg_time, total_time, 'finished'), end='')

		# 'done' notification
		self.log('\n')
		self.log('done with {} sampling steps after {:.2f} seconds'.format(sample_steps, total_time))

		return self._samples

	def append_sample(self, sample) -> str:
		"""Append `sample` vector into current list of sample vectors, `self.samples`.

		The `sample` will only be appended if:

		- all constraints are satisfied and
		- if the `append_criterion` is satisfied

		:param sample: sample vector of the constraint problem which is tested to be appended to the current list of `self.samples` vectors
		:return: message indicating append status, either success: "appending"; or the reason for rejection.
		"""

		if self.constraints.apply(sample):   # check if all constraints are satisfied
			if self.append_criterion(sample):  # check if other criteria are satisfied (e.g. diversity)
				self._samples.append(sample)     # append sample
				return "appending"
			return "atol rejection"  # rejection
		return "constraints"  # rejection

	def append_criterion(self, sample) -> bool:
		"""Diversity criterion for the constraint sampling problem:

		the minimum Eucledian distance of the `sample` vector to all
		other vectors in the current sample list needs to be at least
		than `self.atol`

		:param sample:
		:return: True if all conditions are satisfied, Falso otherwise
		"""
		if len(self._samples) == 0:
			return True

		min_distance = np.linalg.norm(np.asarray(self._samples) - sample, axis=-1).min()

		return min_distance >= self.atol

	def dump_samples(self):
		"""Writes list of drawn sample vectors into `self.output_file`:

		- space delimited within the vector
		- one vector per line
		"""

		# generate file tree to output file (if not exists)
		path = os.path.dirname(self.output_file)
		if not path == '' and not os.path.exists(path):
			try:
				os.makedirs(path)
			except OSError:  # Guard against race condition
				raise IOError('could not create path <{}>'.format(path))

		# write log message and write data to file
		self.log('writing results to output file <{}>'.format(self.output_file))
		np.savetxt(fname=self.output_file, X=np.asarray(self.samples), delimiter=' ')

	@classmethod
	def main(cls, input_file, output_file, n_results, use_example=False, **kwargs) -> list:
		"""Class method which performs sampling of `n_results` vectors on constraints given by `input_file`;
		 writes results to `output_file` (space delimited within the vector, one vector per line)

		:param input_file: Path to Citrine-challenge input file according to `prompt.pdf`; see also README
		:param output_file: Path to output file which is generated, containing a list of `n_results` sampled vectors satisfying the constraints defined by the `input_file`.
		:param n_results: `n_results` of vectors to sample (length of result list)
		:param use_example: Use the example given by the <input_file> during sampling, defaults to False
		:param kwargs: Optional or additional keyword arguments passed on to Sampler constructor (meant for inheriting classes).
		:return: List of sampled vectors satisfying constraints defined by the `input_file` (and additional diversity criteria provided by the `cls.append_criterion` method)
		"""

		# create sampler instance for given input_file, output_file, number of results and other kwargs
		sampler = cls(input_file=input_file, output_file=output_file, n_results=n_results, **kwargs)

		# perform sampling
		if use_example:
			results = sampler.sample(initial_samples=[sampler.constraints.get_example()])
		else:
			results = sampler.sample()

		# dump results to <output_file>
		sampler.dump_samples()

		# return results
		return results
