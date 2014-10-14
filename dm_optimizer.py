from __future__ import print_function

import scipy.optimize as sopt
import scipy
import numpy as np
from numpy.linalg import norm
from copy import copy

import sortedcontainers as sl
from functools import total_ordering
from itertools import imap, islice

import sys

if int(scipy.__version__.split(".")[1]) < 14:
    from scipy.optimize.optimize import Result as OptimizeResult
else:
    from scipy.optimize import OptimizeResult

class BestMinimumException(Exception):
    def BestMinimumException(*args, **kwargs):
        super(*args, **kwargs)

@total_ordering
class value_box:
    """ A silly class that represents points (y, x) compared on the basis of
        their y-components only. """
    def __init__(self, (y, x)):
        self.y = y
        self.x = x

    def unbox(self):
        return (self.y, self.x)

    def __lt__(self, other):
        return self.y < other.y

    def __eq__(self, other):
        return self.y == other.y

class dm_optimizer:
    """ An unconstrained global optimizer using the difference map-based
        algorithm.

        Attributes:
            max_iterations (int):
                the maximum number of iterations to perform before aborting
                minimization.
            tolerance (float):
                the threshold above which we consider two points to be
                distinct.
            verbosity (int):
                controls debug output; higher numbers = more messages.
                Special string 'any' will cause everything to be logged.
            fun (callable):
                the objective function.
            nfev (int):
                total number of function evaluations.
            vals (ordered list of value_box objects):
                the local minima collected along the way, in ascending order
                of score.
            valsi (list of value_box objects):
                the local minima collected along the way, in order of
                discovery.
            lpos (list of positions of the iterate):
                the path the optimizer has followed along the objective
                function.
            callback (callable):
                a function called at the end of every iteration.
                Passed to it is a reference to the optimizer.
            minimizer_kwargs (dict):
                keyword arguments passed to the local minimizer.
            logfile (file handle):
                where debug messages should be logged. (Default: sys.stderr)

        Notes:
            vals is stored as a sorted list, since we frequently just need
            to get the n smallest values.  It is a list of value_box
            objects, which are just wrappers around tuples, sorted according
            to their first element.  to their first element. In order to
            maintain the order in which minima are discovered, we have
            valsi, in which value boxes are stored in the order we find
            them. The number of function evaluations is accounted for by
            only calling it indirectly through a method that updates a
            running count throughout the execution of the optimizer.
        """

    def __init__(self, fun, max_iterations=2500, target=None, constraints=[],
            nonsatisfaction_penalty=0, tolerance=0.000001, verbosity=0,
            logfile=sys.stderr, callback=None, minimizer_kwargs={},
            stepscale_constant=0.5):
        self._fun                       = fun
        self.max_iterations             = max_iterations
        self.constraints                = constraints
        self.nonsatisfaction_penalty    = nonsatisfaction_penalty
        self.tolerance                  = tolerance
        self.verbosity                  = verbosity
        self.logfile                    = logfile
        self.minimizer_kwargs           = minimizer_kwargs
        self.callback                   = callback
        self.stepscale_constant         = stepscale_constant

    def logmsg(self, priority, *args, **kwargs):
        """ Forward all arguments to print if the given priority is less than
            or equal to the verbosity of the optimizer. The file keyword
            argument of print is set to the logfile.

            Arguments:
                priority (int):
                    if this value is less than `verbosity`, all other
                    arguments are forwarded to print.
            """
        if (isinstance(self.verbosity, str) and self.verbosity == "any") \
                or priority == self.verbosity:
            print("%" + str(priority) + "%", *args,
                    file=self.logfile, **kwargs)

    def fun(self, x):
        """ Evaluate the objective function at the given point and increment
            the running count of function calls. """
        self.nfev += 1
        return self._fun(x)

    def evalf(self, x):
        """ Evaluate the objective function at the given point and apply the
            nonsatisfaction penalty times the number of constraints not
            satisfied.
            """
        return self.fun(x) + sum(
                [self.nonsatisfaction_penalty
                    for constraint in self.constraints if not constraint(x)])

    def local_min(self, x):
        """ Find the local minimum nearest to the given point. """
        res = sopt.minimize(self.fun, x, **self.minimizer_kwargs)
        # it is unnecessary to manually increase self.nfev since self.fun
        # already does this for us.  still, we must manually increase
        # self.njev or self.nhev since minimize might use the Jacobian or
        # Hessian
        if 'njev' in res.keys():
            self.njev += res.njev
        if 'nhev' in res.keys():
            self.nhev += res.nhev
        return res.x

    def calculate_step_scale(self, destination):
        """ Calculate the step scale to use in the direction of a given
            destination in the search space. """
        return self.stepscale_constant

    def step_toward(self, destination):
        """ Calculate a step toward a given destination using the standard
            stepscale calculation method.  If the destination's
            function-value is less good than the current one, the direction
            is reversed.
            """
        stepdir = (1 if destination.y <= self.fv else -1) * \
                (destination.x - self.pmin)
        return self.calculate_step_scale(destination) * stepdir

    def get_best_minimum(self):
        """ Get the best past minimum that is at least a certain distance
            away from the current minimum, stored in pmin. """
        for m in self.vals:
            if norm(self.step_toward(m)) >= self.tolerance:
                return m
        raise BestMinimumException("There are no past minima.")

    def step_to_best_minimum(self):
        """ Calculate a step towards the best past minimum. For what "best"
            means, consult the documentation of `get_best_minimum`. Note
            that the step is merely calculated, not taken. For that, pass the
            result of this method to `take_step`.
            """
        return self.step_toward(self.get_best_minimum())

    def fv_after_step_from(self, origin, step):
        return self.evalf(origin + step)

    def fv_after_step_from_iterate(self, step):
        """ Evaluate the score of the objective function after
            hypothetically taking the given step from the iterate. """
        return self.fv_after_step_from(self.nx1, step)

    def fv_after_step_from_minimum(self, step):
        """ Evaluate the score of the objective function after
            hypothetically taking the given step from the minimum. """
        return self.fv_after_step_from(self.pmin, step)

    def all_possible_steps(self):
        return map(lambda x: self.step_toward(x), self.vals)

    def best_of_steps(self, steps):
        return min(steps, key=self.fv_after_step_from_minimum)

    def best_possible_step(self):
        return self.best_of_steps(self.all_possible_steps())

    def average_minima_spread(self):
        """ For each minimum, calculate its average distance to all the other
            minima, and average these averages.  This gives a measure of how
            much of the search space was explored.
            """
        totals = []
        for (i, mini) in enumerate(islice(self.vals, 0, len(self.vals)-1)):
            norms = map(
                    lambda minj: norm(mini.x - minj.x),
                    islice(self.vals, i+1))
            totals.append(sum(norms) / len(norms))
        return sum(totals) / len(totals)

    def take_step(self, step):
        self.nx1 += step
        self.step = step

    def minimize(self, x0, x1):
        """ Perform the minimization given two initial points.

            Arguments:
                x0, x1 (iterable):
                    the initial points.

            Returns:
                res (OptimizeResult):
                    An object describing the result of the optimization. See
                    the SciPy's documentation of this class.
            """
        # Ensure that our lists are empty.
        self.vals                       = sl.SortedList()
        self.valsi                      = []
        self.lpos                       = []

        # and that our counters are zero
        self.nfev       = 0
        self.njev       = 0
        self.nhev       = 0
        self.iteration  = 0

        # and the some other things are given zero initial values
        self.step       = []

        # Get the first minimum
        x0min = self.local_min(x0)
        f0 = self.evalf(x0min)

        self.vals.add(value_box( (f0, x0min) ))
        self.valsi.append(value_box( (f0, x0min) ))

        self.nx1 = x1
        self.fv = f0
        self.pmin = x0min
        self.lpos = [(self.evalf(self.nx1), self.nx1)]

        # prepare the optimization result
        res = OptimizeResult()
        res.message = []

        self.logmsg(2, "Entering loop.")
        try:
            for self.iteration in xrange(1, self.max_iterations + 1):
                self.logmsg(2, "Guess point for this iteration:", self.nx1)
                self.pmin = self.local_min(self.nx1)
                # the function value at the local minimum for this iteration
                self.fv = self.evalf(self.pmin)
                self.logmsg(6, "Minimum for this iteration f", self.pmin,
                        " = ", self.fv, sep='')

                # add the new position to the list of past positions
                self.lpos.append(
                        (self.evalf(self.nx1), copy(self.nx1)))

                if self.callback is not None:
                    if self.callback(self):
                        res.message.append(
                                "Callback function requested termination.")
                        break

                try:
                    self.take_step(self.step_to_best_minimum())
                except BestMinimumException as e:
                    res.message.append(str(e))
                    break

                self.logmsg(2, "Took step ", self.step)

                self.vals.add(value_box( (self.fv, copy(self.pmin)) ))
                self.valsi.append(value_box( (self.fv, copy(self.pmin)) ))
                map(lambda x: self.logmsg(2, x.unbox()), self.vals[-3:])

                if norm(self.step) < self.tolerance:
                    res.message.append("Fixed point found.")
                    break
            else:
                res.message.append("Maximum number of iterations reached.")

        except Exception as e:
            self.logmsg(-1, "An error occurred while optimizing:", e)
            raise e

        # we reach this point if no exceptions have been raised and
        # max_iterations has been exhausted
        self.lpos.append(min(self.vals, key=lambda v: v.y).unbox())
        self.fv, self.pmin = self.lpos[-1]

        res.nit     = self.iteration
        res.success = True
        res.status  = 1
        res.x       = self.pmin
        res.fun     = self.fv
        res.njev    = self.njev
        res.nfev    = self.nfev
        res.lpos    = self.lpos
        res.opt     = self
        return res

def minimize(f, x1, x2, **kwargs):
    """ Construct a dm_optimizer object and run its minimize method with the
        given initial points.  The result is packaged into a scipy
        OptimizeResult object.

        Arguments:
            f       -- the objective function.
            x1, x2  -- the initial points.
            kwargs  -- passed to the constructor of dm_optimizer

        Returns:
            A SciPy OptimizeResult object populated with typical data. Refer
            to the scipy documentation on OptimizeResult for more
            information.
        """
    optimizer = dm_optimizer(f, **kwargs)

    #return optimizer.minimize(x1, x2)
    try:
        res = optimizer.minimize(x1, x2)
    except Exception as e:
        res = OptimizeResult()
        res.message = ["Exception!", str(e)]
        res.status  = 2
        res.success = False
        res.nfev    = optimizer.nfev
        res.njev    = optimizer.njev
        # we'll put the broken optimizer in here in case we need it to
        # recover some failure information
        res.opt = optimizer

    try:
        optimizer.logmsg(1, "Exit message(s):", *res.message)
    except Exception as e:
        print("Fatal error: could not write exit messages to logfile.")
    return res
