from citrine_challenge import Constraint
import numpy as np


# Due to scipy optimization division by zero can occur for eval(ineq_expr) or eval(expr)
# in `citrine_challenge.SamplerConstraints.violate`, showing up in Runtime Warnings.
# (The scipy optimizer tests all boundaries, i.e. |x| > 0. would also need to be a constraint)
# We ignore warnings for now.
np.seterr('ignore')


class SamplerConstraint(Constraint):
	"""Constraints loaded from a file. Inherits `citrine_challenge.Constraint`
	and extends its functionality by the `violation` and `constraint` methods,
	which allows to numerical evaluate the violation of each constraint.
	"""

	def __init__(self, fname):
		"""
		Construct a Constraint object from a constraints file,

		:param fname: Name of the file to read the SamplerConstraint from (string)
		"""

		# call parent constructor
		Constraint.__init__(self, fname)

		with open(fname, "r") as f:
			lines = f.readlines()

		# Run through the rest of the lines and compile the constraints as functionals
		self.exprs_ineq = []
		for i in range(2, len(lines)):
			# support comments in the first line
			if lines[i][0] == "#":
				continue

			# extract the left hand side of the equations g(x) >= 0
			# this gives us the numerical value of constraint violation
			inequality = lines[i].split('>=')[0].strip()

			self.exprs_ineq.append(compile(inequality, "<string>", "eval"))

		return

	def violation(self, x):
		"""
		Apply the constraints to a vector, returning the numerical value
		of all violations or 0 if all constraints are satisfied

		:param x: list or array on which to evaluate the constraints
		:return: current violation of all constraints
		"""
		violations = 0

		for ineq_expr in self.exprs_ineq:
			evaluation = eval(ineq_expr)
			if evaluation < 0.:
				violations -= evaluation

		return violations

	def constraint(self, x, i=None):
		"""
		Apply the constraint `i` to a vector, returning the numerical value

		:param x: list or array on which to evaluate the constraints
		:param i: index of constraint
		:return:
		"""
		return eval(self.exprs_ineq[i])
