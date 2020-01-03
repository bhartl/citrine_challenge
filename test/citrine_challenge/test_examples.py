import unittest2
from time import time
import numpy as np


class TestExample(unittest2.TestCase):
	"""Demonstrative Unittest to test the performance of `citrine_challenge.AdaptiveSampler`"""

	def setUp(self):
		"""setup path variables to the example files and output files"""
		self.input_file_example = "test/files/example.txt"
		self.output_file_example = "test/dat/citrine_challenge/example.txt"

		self.input_file_mixture = "test/files/mixture.txt"
		self.output_file_mixture = "test/dat/citrine_challenge/mixture.txt"

		self.input_file_alloy = "test/files/alloy.txt"
		self.output_file_alloy = "test/dat/citrine_challenge/alloy.txt"

		self.input_file_formulation = "test/files/formulation.txt"
		self.output_file_formulation = "test/dat/citrine_challenge/formulation.txt"

	def sample(self, sampler, verbose=False, time_constraint_in_sec=300):
		"""test sample method for a given sampler:

		- generate results for constraint problem,
		- check output_file generation
		- check if reloaded outputs still satisfy constraints
		- track time (challenge limit <= 5 min)

		:param sampler: `citrine_challenge.BaseSampler` instance (or derivative thereof)
		:param verbose: boolean value if supportive information is printed (True) or not (False)
		:param time_constraint_in_sec: sampling should be performed below this time threshold
		"""

		if verbose:
			heading = '--- test <{}> ---'.format(sampler.input_file)
			print('-'*len(heading))
			print(heading)

		start_time = time()
		results = sampler.sample(return_info=True)
		end_time = time()
		sample_time = end_time - start_time

		if verbose: print('--- dump to <{}> ---'.format(self.output_file_example))
		sampler.dump_samples()

		if verbose: print('--- check if output results are the same as the ones from the method')
		results_output = np.loadtxt(sampler.output_file)
		self.assertTrue(np.allclose(results_output, results))

		if verbose: print('--- check if all constraints apply')
		for result, read_results in zip(results, results_output):
			constraints_apply_original = sampler.constraints.apply(result)
			constraints_apply_output = sampler.constraints.apply(read_results)

			# if verbose: print(result, 'constraints apply:', constraints_apply_original)

			self.assertTrue(constraints_apply_original)
			self.assertTrue(constraints_apply_output)

		if verbose: print('---')

		if verbose: print('--- check if sampling lies within the given time constraint')
		self.assertTrue(sample_time <= time_constraint_in_sec)
		if verbose:	print('-' * len(heading))

	def test_mixture(self, n_results=1000, verbose=True, log_interval=100):
		"""Perform sampling on the mixture.txt input_file"""

		# BaseSampler using random samples, still works for the mixture task
		from citrine_challenge import BaseSampler

		# Adaptive Sampler using Scipy minimization and de-correlation
		from citrine_challenge import AdaptiveSampler

		for Sampler, name in zip([BaseSampler, AdaptiveSampler], ['-base_sampler', '-adaptive_sampler']):
			# use different output_files for BaseSampler and AdaptiveSampler
			output_file = self.output_file_mixture.replace('.txt', '') + name + '.txt'

			# generate Sampler instance
			sampler = Sampler(
				input_file=self.input_file_mixture,
				output_file=output_file,
				n_results=n_results,
				log_interval=log_interval,  # log interval for solver steps
				verbose=verbose,            # verbosity defined by parameter
			)

			# call unittest routine above
			self.sample(sampler, verbose=verbose)

	def test_example(self, n_results=1000, verbose=True, log_interval=100):
		"""Perform sampling on the example.txt input_file"""

		# BaseSampler using random samples, still works for the example task
		from citrine_challenge import BaseSampler

		# Adaptive Sampler using Scipy minimization and de-correlation
		from citrine_challenge import AdaptiveSampler

		for Sampler, name in zip([BaseSampler, AdaptiveSampler], ['-base_sampler', '-adaptive_sampler']):
			# use different output_files for BaseSampler and AdaptiveSampler
			output_file = self.output_file_example.replace('.txt', '') + name + '.txt'

			# generate Sampler instance
			sampler = Sampler(
				input_file=self.input_file_example,
				output_file=output_file,
				n_results=n_results,
				log_interval=log_interval,  # log interval for solver steps
				verbose=verbose,            # verbosity defined by parameter
			)

			# call unittest routine above
			self.sample(sampler, verbose=verbose)

	def test_formulation(self, n_results=1000, verbose=True, log_interval=100):
		"""Perform sampling on the formulation.txt input_file"""

		# BaseSampler using random samples, still works for the formulation task
		from citrine_challenge import BaseSampler

		# Adaptive Sampler using Scipy minimization and de-correlation
		from citrine_challenge import AdaptiveSampler

		for Sampler, name in zip([BaseSampler, AdaptiveSampler], ['-base_sampler', '-adaptive_sampler']):
			# use different output_files for BaseSampler and AdaptiveSampler
			output_file = self.output_file_formulation.replace('.txt', '') + name + '.txt'

			# generate Sampler instance
			sampler = Sampler(
				input_file=self.input_file_formulation,
				output_file=output_file,
				n_results=n_results,
				log_interval=log_interval,  # log interval for solver steps
				verbose=verbose,            # verbosity defined by parameter
			)

			# call unittest routine above
			self.sample(sampler, verbose=verbose)

	def test_alloy(self, n_results=1000, verbose=True, log_interval=100):
		"""Perform sampling on the alloy.txt input_file"""

		# this won't work in a reasonable time for the alloy problem:
		#	from citrine_challenge import BaseSampler as Sampler

		from citrine_challenge import AdaptiveSampler as Sampler

		sampler = Sampler(
			input_file=self.input_file_alloy,
			output_file=self.output_file_alloy,
			n_results=n_results,
			log_interval=log_interval,  # log interval for solver steps
			verbose=verbose,            # verbosity defined by parameter
			n_correlated_samples=20,       # using 42 random samples in the correlation matrix turned out to be reasonable wrt. performance
			normalize_pca_threshold=0.1,  # a threshold of 0.09 turned out to reasonable wrt. performance
		)

		self.sample(sampler, verbose=verbose)
