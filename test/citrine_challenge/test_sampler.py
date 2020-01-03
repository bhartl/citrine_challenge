import unittest2
import os


class TestSampler(unittest2.TestCase):
	"""Demonstrative Unittests for the BaseSampler class"""

	def setUp(self):
		"""setup path variables to the example files and output files"""

		self._input_files = [
			"test/files/example.txt",
			"test/files/alloy.txt",
			"test/files/formulation.txt",
			"test/files/mixture.txt",
		]

		self._output_prefix = "test/dat/citrine_challenge/TestSampler"

	def test_init(self):
		"""test initialization of `citrine_challenge.BaseSampler`"""

		# check import
		from citrine_challenge import BaseSampler as Sampler

		# use positional arguments
		Sampler("test/files/example.txt", "some_output_file.txt", 0)

		# use keyword-arguments
		Sampler(
			input_file="test/files/example.txt",
			output_file=os.path.join(self._output_prefix, "some_output_file.txt"),
			n_results=0
		)

		# check constructor has mandatory arguments
		with self.assertRaises(TypeError):
			Sampler()

		# check constructor has mandatory arguments
		with self.assertRaises(AssertionError):
			Sampler("test/files/example.txt", "some_output_file.txt", "not a number")

		# check if assertion is raised if `n_results`
		with self.assertRaises(AssertionError):
			Sampler("test/files/example.txt", "some_output_file.txt", 9.2)

		# check if assertion is raised if `output_file` is not a string
		with self.assertRaises(AssertionError):
			Sampler("test/files/example.txt", 1234, 9)

		# check if assertion is raised if File not exists
		with self.assertRaises(AssertionError):
			Sampler("this is not a file", "some_output_file.txt", 0)

	def test_properties(self):
		"""check if `citrine_challenge.BaseSampler` properties are set accordingly"""

		from citrine_challenge import BaseSampler

		# use example files defined in the self.setUp routine defined
		# by the unittest framework (these may be used frequently)
		input_files = self._input_files

		output_files = [
			os.path.join(self._output_prefix, "test_properties/some_output_file.txt"),
			os.path.join(self._output_prefix, "test_properties/another_output_file.txt"),
			os.path.join(self._output_prefix, "test_properties/another_output_file.txt")
		]

		n_results = [0, 1, 10, 100]

		# make a grid of all compinations of input arguments
		# note: one could use itertools.product(*args) here
		for input_file  in input_files:
			for output_file in output_files:
				for n in n_results:

					# use keyword-arguments
					sampler = BaseSampler(input_file=input_file, output_file=output_file, n_results=n)

					# check if properties are set accordingly for each realization of
					# the arguments above
					self.assertEqual(input_file, sampler.input_file)
					self.assertEqual(output_file, sampler.output_file)
					self.assertEqual(n, sampler.n_results)

					# one could also check, if the
					self.assertTrue(sampler._input_file is sampler.input_file)
					self.assertTrue(sampler._output_file is sampler.output_file)
					self.assertTrue(sampler._n_results is sampler.n_results)

					# check assertion clause for n_results
					with self.assertRaises(AssertionError):
						sampler.n_results = "not a number"
					# value should be unchanged
					self.assertEqual(n, sampler.n_results)

					# one could also check if new values can be assigned correctly
					sampler.n_results = 1234
					self.assertEqual(1234, sampler.n_results)
					self.assertEqual(1234, sampler._n_results)

	def test_example(self):
		"""check if `citrine_challenge.BaseSampler` reads constraints
		correctly and examples satisfy these constraints"""

		from citrine_challenge import BaseSampler

		input_files = self._input_files

		output_files = [
			os.path.join(self._output_prefix, "test_example/some_output_file.txt"),
		]

		n_results = [1]

		# make a grid of all compinations of input arguments
		# note: one could use itertools.product(*args) here
		for input_file  in input_files:
			for output_file in output_files:
				for n in n_results:

					# use keyword-arguments
					sampler = BaseSampler(input_file=input_file, output_file=output_file, n_results=n)

					# check constraints read correctly and example satisfies constraint
					example = sampler.constraints.get_example()
					self.assertTrue(sampler.constraints.apply(example))

					# todo: sample configurations which don't satisfy constraints and test with lambdas
