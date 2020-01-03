from scipy.optimize import minimize
from citrine_challenge import BaseSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# Due to scipy optimization division by zero can occur for eval(ineq_expr) or eval(expr)
# in `citrine_challenge.SamplerConstraints.violate`, showing up in Runtime Warnings.
# (The scipy optimizer tests all boundaries, i.e. |x| > 0. would also need to be a constraint)
# We ignore warnings for now.
np.seterr('ignore')


class AdaptiveSampler(BaseSampler):
	""" `Scipy` based Sampler, derived from `BaseSampler`, which uses `scipy.optimize.minimize`
	(i) to find feasible regions by minimizing the negative constraint violations from `SamplerConstraint`
	(ii) and to adaptively minimize correlations in the drawn samples.

	The correlation minimization is performed on normalized data samples
	using  `scikit-learn`'s `StandardScaler` (removing the mean and scaling
	to unit variance) and `PCA` (reduce dimensionality).

	The following `BaseSampler` methods are overwritten:

	- `draw_sample`
	- `append_sample`
	- `append_criterion`

	Note: In order to make this Sampler truly adaptive, one could overwrite the `BaseSampler.converged` property.
	A convergence criteria could be to minimize the correlation of the entire samples-list below a certain
	threshold. But for the current implementation this is too time-consuming and also needs more conceptual
	considerations.
	"""

	def __init__(self, *args, n_correlated_samples=100, normalize_pca_threshold=0.08, max_draw_steps=1000, atol=1e-4, **kwargs):
		""" Construct an AdaptiveSampler object from a constraints file,

		:param args: Positional arguments, passed to `BaseSampler` constructor
		:param n_correlated_samples: Number of samples considered in minimizing correlations (positive integer)
		:param max_draw_steps: Integer which controls the maximum number of attempts to draw new examples (in order to
		                       avoid endless loops)
		:param normalize_pca_threshold:  Threshold of `PCA` variance which are considered as important in
		                                 order to reduce the dimensionality of the constraint problem (semi positive float)
		:param atol: Positive float value which controls the minimum Eucledian distance which all sample vectors must exhibit
		             before being added to the samples list in normalized space
		:param kwargs: Keyword arguments, passed to `BaseSampler` constructor
		"""
		BaseSampler.__init__(self, *args, max_draw_steps=max_draw_steps, atol=atol, **kwargs)

		self._n_correlated_samples = None                  # number of samples considered in minimizing correlations
		self.n_correlated_samples = n_correlated_samples

		self._n_correlated_offset = 0                      # offset for initial samples: refilled with de-correlated samples

		self._normalized = False                           # helper, if samples have been normalized for de-correlation
		self._normalized_samples = []                      # list of all normalized samples for de-correlation
		self._normalized_representatives = []              # list of `n_correlated_samples` normalized samples for de-correlation

		self._normalize_scaler = None                      # scikit-learn StandardScaler helper
		self._normalize_pca = None                         # scikit-learn Principal Component Analysis helper
		self._normalize_pca_threshold = None               # threshold of PCA variance which are considered during de-correlation steps
		self.normalize_pca_threshold = normalize_pca_threshold
		self._normalized_leading = [True] * self.constraints.n_dim  # leading components of normalized vectors which are considered important

		# list of inequality constraints for in the correlation minimization from SamplerConstraint violations
		self._constraints_list = list([
			{'type': 'ineq', 'fun': lambda x: self.constraints.constraint(x, i=i)}
			for i in range(len(self.constraints.exprs_ineq))
		])

	@property
	def normalize_pca_threshold(self) -> float:
		"""Get threshold of PCA variance which are considered during de-correlation steps"""
		return self._normalize_pca_threshold

	@normalize_pca_threshold.setter
	def normalize_pca_threshold(self, normalize_pca_threshold):
		"""Set threshold of PCA variance which are considered during de-correlation steps

		:param normalize_pca_threshold: semi positive float
		"""
		assert normalize_pca_threshold >=0, "The PCA threshold must be larger or equal to 0."
		self._normalize_pca_threshold = normalize_pca_threshold

	@property
	def n_correlated_samples(self) -> int:
		"""Get number of samples considered in minimizing correlations"""
		return self._n_correlated_samples

	@n_correlated_samples.setter
	def n_correlated_samples(self, n_correlated_samples):
		"""Set number of samples considered in minimizing correlations

		:param n_correlated_samples: positive int
		"""
		assert int(n_correlated_samples) > 0, "The number of correlated samples must be larger than 0."
		self._n_correlated_samples = int(n_correlated_samples)

	def draw_sample(self, seed=None, **kwargs) -> tuple:
		"""Draw sample using `scipy.optimize.minimize`
		(i) to find feasible regions by minimizing the negative constraint violations and
		(ii) to minimize correlations in the drawn samples.

		Overwrites `BaseSampler.draw_sample` method.

		:param seed: seed for random number generator, defaults to None
		:param kwargs: possible kwargs for inheriting classes
		:return: (sample array, number of draw attempts)
		"""

		done = False
		sample = None
		attempts = 0

		# search until new candidate sample is found (or self._max_draw_steps is reached)
		while not done:

			# draw random sample on n_dim hypercube and evaluate next local minimum
			sample = self.minimize_constraints(seed)

			# wait initialization period before trying to de-correlate newly drawn samples
			if self.constraints.apply(sample) and len(self.samples) >= self._n_correlated_samples:
				try:
					# de-correlate sample by minimizing its correlations to the other, already drawn samples
					sample = self.minimize_correlations(sample)

				except ValueError:  # protect against bad PCA transformation
					attempts += 1
					continue

			# increase status counter
			attempts += 1

			# evaluate, if constraints are satisfied
			done = self.constraints.apply(sample)

			#  check if max_draw_steps are exceeded
			if attempts > self._max_draw_steps and not done:
				break

		return sample, attempts

	def minimize_constraints(self, seed=None, x0=None, **kwargs) -> np.ndarray:
		"""Draw sample on unit-hypercube at random (if no `x0` is provided) and evaluate next
		local minimum by use of the `scipy-optimize.minimize` implementation of `L-BFGS-B`
		minimization of the negative constraint violation (`sum(-g(x) * theta(-g(x))`) of a
		given sample.

		:param seed: seed for random number generator, defaults to None
		:param x0: list or array, initial configuration which is to be optimized defaults to None,
		           if not provided random `x0` is drawn
		:param kwargs: kwargs to be passed on to `scipy-optimize.minimize`
		               (`x0`, `fun`, `bounds` and `method=L-BFGS-B` already specified),
									 optional additional parameters for optimizer
		:return: sample vector minimized for negative constraint violation
		"""

		# initialize random number generator
		if seed is not None:
			np.random.seed(seed)

		# draw random sample on n_dim hypercube as initial configuration for scipy minimization
		if x0 is None:
			x0 = np.random.rand(self.constraints.n_dim)

		# call scipy optimization
		result = minimize(
			x0=x0,                                      # initial configuration
			fun=self.constraints.violation,             # functional to be optimized (negative constraint violations)
			method='L-BFGS-B',                          # optimization method
			bounds=[(0., 1.)]*self.constraints.n_dim,   # boundaries for sample, each element must be between 0 and 1
			**kwargs                                    # optional additional parameters for optimizer
		)

		# here we could also check for success of the optimization
		# since our goal is to satisfy the constraints given by the input_file
		# we simple return the resulting vector
		sample = result.x

		return sample

	def minimize_correlations(self, x0, **kwargs) -> np.ndarray:
		"""Minimize the correlation of a new `sample` vector and the current list of samples previously evaluated
		(or better of `n_correlated_samples` randomly chosen samples thereof -- for better performance).

		The `sample` as well as the `n_correlated_samples` randomly chosen representatives of the
		current list of samples are normalized using `scikit-learn`'s `StandardScaler`.

		To reduce the dimensionality of the constraint problem (and again for performance reasons)
		we subsequently transform the problem into the frame of the leading `Principal Components`
		of the normalized samples which's variances are larger than the specified `normalize_pca_threshold`.

		:param x0: list or array, initial configuration which is to be optimized
		:param kwargs: kwargs to be passed on to `scipy-optimize.minimize`
		               (`x0`, `fun`, `bounds` and `method=SLSQP` already specified),
									 optional additional parameters for optimizer
    :return: optimization result as numpy array
		"""
		assert self.constraints.apply(x0), "Initial sample needs to satisfy constraints."

		# normalize samples
		self.normalize()

		# minimize correlation
		result = minimize(
			x0=x0,
			fun=self.sample_correlation,
			method='SLSQP',
			bounds=[(0., 1.)]*self.constraints.n_dim,
			constraints=self._constraints_list,
			**kwargs
		)

		# we again, don't check for successful optimization
		sample = result.x

		# but we check if sample satisfies the constraints
		if not self.constraints.apply(sample):
			# constraints may be violated, 'SLSQP' is not 100% bulletproof here,
			# so we have to manually help the sample back to a feasible region.
			sample = self.minimize_constraints(x0=sample)

		return sample

	@property
	def normalized(self):
		"""Boolean if sample has been normalized (via `self.normalize()`)"""
		return self._normalized

	def normalize(self):
		"""Normalize samples using `StandardScaler` and `PCA`

		The current list of samples are normalized using `scikit-learn`'s `StandardScaler`
		to de-correlate the data (removing the mean and scaling to unit variance).
		Some problems may be highly constraint thus the data will occupy
		a certain subspace of the entire n dimensional hyper cube.

		To reduce the dimensionality of the constraint problem (and for performance reasons)
		we subsequently transform the problem into the frame of the leading `Principal Components`
		of the normalized samples which's variances are larger than the specified `normalize_pca_threshold`.

		From the transformed samples `n_correlated_samples` representatives are chosen at random, which
		can subsequently be used to evaluate the correlation with a given sample.

		:return: list of `n_correlated_samples` normalized sample vectors
		"""

		# Due to the constraint problem the data in self.samples may be highly correlated
		# we try to de-correlate the samles here: removing the mean and scaling to unit variance
		# (maybe diffusion maps may be an alternative for some geometries)
		self._normalize_scaler = StandardScaler()
		self._normalized_samples = self._normalize_scaler.fit_transform(self.samples)

		# for PCA to work we need at least `n_dim` samples
		if len(self._normalized_samples) >= self.constraints.n_dim:
			# PCA instance
			self._normalize_pca = PCA(self.constraints.n_dim)

			# find transofmration using normalized samples
			self._normalized_samples = self._normalize_pca.fit_transform(self._normalized_samples)

			# use leading principal components to reduce dimensions (variances must be larger than threshold)
			self._normalized_leading = (self._normalize_pca.explained_variance_ratio_ >= self._normalize_pca_threshold)
			if not any(self._normalized_leading):  # keep at least 2 dimensions
				self._normalized_leading[0:2] = True
			self._normalized_samples = self._normalized_samples[:, self._normalized_leading]

		# pick n_correlated_samples at random if size of self.samples exceeds n_correlated_samples
		if len(self.samples) > self._n_correlated_samples:
			random_representatives = np.random.choice(np.arange(len(self.samples)), self._n_correlated_samples, replace=False)
			self._normalized_representatives = self._normalized_samples[random_representatives]
		else:
			self._normalized_representatives = self._normalized_samples

		self._normalized = True

		return self._normalized_representatives

	def normalize_transform(self, sample) -> np.ndarray:
		"""Normalization transformation to single samples or list of samples,
		(i) `StandardScaler` and (ii) `PCA` transformations are subsequently performed,
		only the leading `self._normalized_leading` are considered in the normalized vectors.

		:param sample: Single sample vector or list of sample vectors which are to be transformed
		:return: Normalized single sample (1d array) or list of samples (2d array)
		"""

		if not self.normalized:
			self.normalize()

		# check, if single sample or list of samples to be transformed
		single_sample = np.ndim(sample) == 1
		if single_sample:
			# scikit-learns StandardScaler requires 2d array for transformation
			sample = [sample]

		# perform StandardScaler transformation according to model defined in `self.normalize()`
		transformed = self._normalize_scaler.transform(sample)

		# if pca transformation is available (if enough data are available) we also perform this transformation
		if self._normalize_pca is not None:
			transformed = self._normalize_pca.transform(transformed)

		# check, if a single sample was requested, ensure 1d array is returned, only consider leading principal components
		if single_sample:
			return transformed[0, self._normalized_leading]

		# return 2d-array of normalized samples, only consider leading principal components
		return transformed[:, self._normalized_leading]

	def sample_correlation(self, sample=None):
		"""evaluates the norm of the correlation coefficients of the representative
		normalized samples and the (not normalized) sample argument is evaluated.
		The functional is regularized by addition of `self.constraints.violation(sample)`
		to support search in feasible regions.

		:param sample: list or array for which correlation with respect to the representative normalized samples is evaluated
		:return: norm of correlation coefficients of normalized `sample` with representative normalized samples
		"""

		# perform normalization on sample
		normalized_sample = self.normalize_transform(sample)

		# evaluate correlation matrix for single normalized_sample and list of normalized samples
		correlations = np.corrcoef(self._normalized_representatives, normalized_sample)

		# relevant correlations are located in last row or line,
		# i.e. correlations of sample with list of samples,
		# the variance being the [-1, -1] element in the matrix (and is omitted).
		correlations = np.linalg.norm(correlations[:-1, -1])

		## optional functional value also additionally maximizing minimum distance:
		## (this needs some more thought)
		# min_distance = np.min(np.linalg.norm(self._normalized_samples - normalized_sample, axis=-1))
		# return correlations - min_distance/self.constraints.n_dim

		return correlations + self.constraints.violation(sample)

	def append_criterion(self, sample):
		"""Diversity criterion for the constraint sampling problem:

		If there is not enough data for normalization we always allow
		the example (these initial examples are eliminated later on).

		If normalization of the data has been performed already,
		we only append the sample, if the minimum Eucledian distance
		of the normalized `sample` vector to all other normalized
		vectors in the current sample list is at least than `self.atol`.
		Note, that we are considering all data here, not only `n_correlated_samples`
		as in the evaluation of `sample_correlation` above.

		:param sample: list or array which is checked for the appending conditions
		:return: True if all conditions are satisfied, False otherwise
		"""

		# get a set of initial samples
		if not self.normalized:
			return True

		normalized_samples = self.normalize_transform(self.samples)  # normalize all samples
		normalized_sample = self.normalize_transform(sample)         # normalize the candidate sample

		# take the minimum normalized distance of the candidate sample to all other samples
		normalized_min_distance = np.linalg.norm(normalized_samples - normalized_sample, axis=-1).min()

		return normalized_min_distance >= self.atol

	def append_sample(self, sample):
		"""Append `sample` vector into current list of
		sample vectors, `self.samples`.

		The `sample` will only be appended if:

		- all constraints are satisfied and
		- if the `append_criterion` is satisfied

		:param sample: list or array, vector sample of the constraint problem
		              which is tested to be appended to the current list of
		              `self.samples` vectors
		:return: message indicating the append status,
		         either initialization stage: "initialize" or "de-correlate";
						 success of appending subsequent samples: "appending";
						 or the reason for rejection.
		"""

		if self.constraints.apply(sample):   # check if all constraints are satisfied
			if self.append_criterion(sample):  # check if other criteria are satisfied (e.g. diversity)
				if len(self.samples) < self._n_correlated_samples:  # check, if we are in the initialization stage
					self._samples.append(sample)                      # -> here normalization is a problem
					return "initialize"

				elif self._n_correlated_offset >= self._n_correlated_samples:  # check, if we are in the "appending" stage
					self._samples.append(sample)                                 # initialization and de-correlation is done
					return "appending"

				else:                                                # here we fill the first self._n_correlated_offset
					self._samples[self._n_correlated_offset] = sample  # with new configurations, since we do not optimize
					self._n_correlated_offset += 1                     # for minimal correlation in the first steps

					return "de-correlate {}".format(self._n_correlated_offset)
			return "atol rejection"  # rejection
		return "constraints"  # rejection

