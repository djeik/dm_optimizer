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

    @staticmethod
    def from_yx(y, x):
        """ Construct a value box from a scalar y and a vector x. """
        return value_box( (y, x) )

    def __init__(self, (y, x)):
        """ Construct a value_box from a tuple (y, x). """
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
            stepscale_constant=0.5, minimum_distance_ratio=0.5):
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
        self.minimum_distance_ratio     = minimum_distance_ratio

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
        """ Find the local minimum nearest to the given point.
            The return value is a value_box.
            """
        res = sopt.minimize(self.fun, x, **self.minimizer_kwargs)
        # it is unnecessary to manually increase self.nfev since self.fun
        # already does this for us.  still, we must manually increase
        # self.njev or self.nhev since minimize might use the Jacobian or
        # Hessian
        if 'njev' in res.keys():
            self.njev += res.njev
        if 'nhev' in res.keys():
            self.nhev += res.nhev
        return value_box.from_yx(res.fun, res.x)

    ###### STEPTAKING STRATEGIES ######
    def _constant_factor_reversing_steptake_strategy(self, origin, destination):
        """ Calculate a step between the two points represented as
            value_box objects by taking the direction to be the one towards
            decreasing values of y, and multiplying by a fixed scale constant
            (self.stepscale_constant)
            """
        stepdir = (1 if destination.y <= origin.y else -1) * \
                (destination.x - origin.x)
        return self.stepscale_constant * stepdir

    def _adaptive_steptake_strategy(self, origin, destination):
        """ Calculate a step between the two points represented as value_box
            objects by taking the vector between then, oriented towards the
            destination, scaled by the percent difference between the values of
            y.
            """
        def fallback():
            self.logmsg(1, "falling back to constant factor reversing "
                    "steptake strategy.")
            return self._constant_factor_reversing_steptake_strategy(
                    origin,
                    destination)

        # If the origin cost is too close to zero, then the scale will get
        # weird. So we fall back to a constant stepscale in that case.
        if origin.y**2 < self.tolerance**2:
            self.logmsg(1, "origin score too close to zero")
            fallback()

        scale = (origin.y - destination.y) / origin.y

        direction = destination.x - origin.x
        step = scale * direction

        if norm(step)**2 < self.tolerance**2:
            self.logmsg(1, "step too small")
            fallback()

        return step
    ###### END STEPTAKING STRATEGIES ######

    def step_toward(self, destination, steptaking_strategy):
        """ Calculate a step from the current local minimum to the given
            destination point using the given steptaking strategy. The current
            local minimum is supplied as the first argument to the steptaking
            strategy.

            A steptaking strategy is a binary callable taking an origin and a
            destination, both as value_box objects. The returned value must be
            a vector, in the same number of dimensions as the search space,
            that is to be added to the iterate.
            """
        return steptaking_strategy(
                self.current_minimum,
                destination)

    @staticmethod
    def _distance_ratio(p1, p2, p3):
        """ Calculates the distance between the p1 and p2 over the distance
            between p2 and p3. This is a helper function for the distance
            ratio distinctness strategy.
            """
        u = norm(p2 - p1)
        v = norm(p3 - p2)
        return (u, v)

    ### DISTINCTNESS STRATEGIES ###
    def _distance_ratio_distinctness_strategy(self, v1, v2):
        """ Determine whether two points are distinct from the point of view of
            the iterate by taking the distance from the iterate to p1 over the
            distance from p1 to p2.
            """
        u, v = dm_optimizer._distance_ratio(self.iterate, v1.x, v2.x)
        return u >= v * self.minimum_distance_ratio

    def _epsilon_threshold_distinctness_strategy(self, v1, v2):
        """ Determine whether two points are distinct from the point of view of
            the iterate by checking that the distance between the two points is
            greater than a fixed threshold called epsilon. This epsilon is
            simply the tolerance (self.tolerance).
            This strategy is fundamentally maladaptive, since it uses a
            predetermined constant threshold, which must be determined per
            objective function.
            """
        return norm(v2.x - v1.x) >= self.tolerance
    ### END DISTINCTNESS STRATEGIES ###

    ### FAILURE STRATEGIES ###
    def _worst_minimum_failure_strategy(self, local, minima):
        """ A failure strategy for the `get_best_minimum` method.
            This strategy returns the worst minimum from the list, in terms of
            score.
            """
        return minima[-1]

    def _best_minimum_failure_strategy(self, local, minima):
        """ A failure strategy for the `get_best_minimum` method.
            This strategy returns the best minimum form the list, in terms of
            score.
            """
        return minima[0]

    def _furthest_minimum_failure_strategy(self, local, minima):
        """ A failure strategy for the `get_best_minimum` method.
            This strategy returns the past minimum that is the furthest away
            from the local minimum, according to their location in the search
            space, without paying attention to the score of the minima.
            """
        return max(minima, key=lambda m: norm(m.x - local.x))

    def _exception_failure_strategy(self, local, minima):
        """ A failure strategy for the `get_best_minimum` method.
            This strategy simply raises a BestMinimumException.
            The minima given as arguments are therefore ignored.
            This will typically cause the termination of the optimizer, except
            in the case of an overarching exception-swallowing strategy.
            """
        raise BestMinimumException("There are no past minima.")
    ### END FAILURE_STRATEGIES ###

    def get_best_minimum(self, distinctness_strategy, failure_strategy):
        """ Get the best distinct past minimum, according to the given
            distinctness strategy. If no minima meet is distinctness criterion,
            then the given failure strategy is invoked.

            The distinctness strategy is a decision procedure taking two
            value_box objects and determining whether they are distinct (True)
            or not (False) in the search space. The current local minimum is
            supplied as the first argument, and a past minimum is supplied as
            the second argument, but this should not be relied on: distinctness
            strategies should be reflexive, obeying the following law.

                D(x, y) == D(y, x) for all x, y
                    where D is a distinctness strategy

            The failure_strategy is a binary callable, given the current minimum
            and the entire list of past minima, sorted in ascending order of
            score. Possible strategies include simply taking the best or worst
            minimum by score, picking one at random, or choosing one according
            to its distance from the current minimum.
            """
        for m in self.vals:
            if distinctness_strategy(self.current_minimum, m):
                return m
        self.logmsg(2, "invoking failure strategy:", failure_strategy.__name__)
        return failure_strategy(self.current_minimum, self.vals)

    def step_to_best_minimum(self):
        """ Calculate a step towards the best past minimum. For what "best"
            means, consult the documentation of `get_best_minimum`. Note
            that the step is merely calculated, not taken. For that, pass the
            result of this method to `take_step`.
            """
        return self.step_toward(
                self.get_best_minimum(
                    self._distance_ratio_distinctness_strategy, # parameterize !
                    self._furthest_minimum_failure_strategy),
                self._constant_factor_reversing_steptake_strategy)

    def take_step(self, step):
        """ Add the given value to the iterate, and set the step attribute to
            that value.
            """
        self.iterate += step
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
        self.vals  = sl.SortedList()
        self.valsi = []
        self.lpos  = []

        # and that our counters are zero
        self.nfev      = 0
        self.njev      = 0
        self.nhev      = 0
        self.iteration = 0

        # and the some other things are given zero initial values
        self.step = []

        # Get the first minimum
        #x0min, f0 = self.local_min(x0)
        self.current_minimum = self.local_min(x0)

        self.vals.add(self.current_minimum)
        self.valsi.append(self.current_minimum)

        self.iterate = x1
        self.lpos = [(self.evalf(self.iterate), self.iterate)]

        # prepare the optimization result
        res         = OptimizeResult()
        res.message = []

        # The main optimizer loop
        self.logmsg(2, "Entering loop.")

        # We bracket the whole optimizer loop in an exception handler, so that
        # we can consider the number of failures. See the except block below.
        try:
            for self.iteration in xrange(1, self.max_iterations + 1):
                self.logmsg(1, "ITERATION:", self.iteration)
                self.logmsg(2, "Guess point for this iteration:", self.iterate)

                # Perform a local minimization at the location of the iterate.
                self.current_minimum = self.local_min(self.iterate)

                self.logmsg(6, "Minimum for this iteration f",
                        self.current_minimum.x, " = ", self.current_minimum.y,
                        sep='')

                # Add the current position of the iterate to the list of past
                # positions, since the next step is to move the iterate.
                self.lpos.append(
                        (self.evalf(self.iterate), copy(self.iterate)))

                # If an iteration hook / callback is installed, we run it now.
                if self.callback is not None:
                    # Callbacks can terminate the optimizer if they return True.
                    if self.callback(self):
                        res.message.append(
                                "Callback function requested termination.")
                        break

                # Run the step-taking procedure.
                # The step-taking procedure is wrapped in an exception handler,
                # since it is possible that the step-taker request termination
                # of the optimizer via an exception.
                try:
                    self.take_step(self.step_to_best_minimum())
                except BestMinimumException as e:
                    res.message.append(str(e))
                    break

                self.logmsg(2, "Took step ", self.step)

                # Record the current local minimum into the sorted list of past
                # minima. The best past minimum will be at index 0. We use
                # value_box objects to enforce the ordering on the y-component
                # only.
                self.vals.add(self.current_minimum)

                # Record the current local minimum into the list of past minima
                # by time. This list presents the minima in the order that they
                # were discovered. This list is useful for making plots, whereas
                # the sorted by score list is useful for step-taking algorithms.
                self.valsi.append(self.current_minimum)

                map(lambda x: self.logmsg(2, x.unbox()), self.vals[-3:])
            else:
                res.message.append("Maximum number of iterations reached.")

        # If an exception occurs, then we would like to log the exception before
        # reraising it to the caller.
        except Exception as e:
            self.logmsg(-1, "An error occurred while optimizing:", e)
            raise e

        # we reach this point if no exceptions have been raised and one of the
        # following criteria are met:
        # 1) max_iterations has been exhausted
        # 2) The callback requested termination of the optimizer.
        # 3) The step length fell below the tolerance.
        # 4) The step-taking procedure requested termination (or failed)

        # The final position is simply taken to be the best minimum found.
        # value_box is compared on y, so a key is not necessary for min
        self.current_minimum = self.vals[0]

        # Prepare the OptimizeResult object
        res.nit     = self.iteration # Number of iterations
        res.success = True # Whether the optimizer completed successfully
        res.status  = 1 # The exit code of the optimizer
        res.x       = self.current_minimum.x # The value of x at the optimum
        res.fun     = self.current_minimum.y # The value of y at the optimum
        res.nfev    = self.nfev # The number of function evaluations
        res.njev    = self.njev # The number of evaluations of the Jacobian
        res.lpos    = self.lpos # The list of positions of the iterate

        # The past minima, in order of discovery, represented as tuples.
        res.valsi   = map(lambda v: v.unbox(), self.valsi)

        # The optimizer itself, in order to extract any other information.
        # If the OptimizeResult needs to be pickled, then having the optimizer
        # inside is undesirable, since it contains unpicklable objects. In that
        # case, pass the result through the sanitize_result utility function.
        res.opt     = self
        return res

def sanitize_result(res):
    """ Clean up an OptimizeResult, so that it only contains picklable
        objects. """
    r = OptimizeResult()

    def copyattr(s):
        setattr(r, s, getattr(res, s))

    def safecopyattr(s):
        try:
            copyattr(s)
        except AttributeError:
            pass # exception-swallowing!

    map(safecopyattr, ["message", "status", "success", "nfev", "njev",
        "lpos", "valsi", "nit", "x", "fun"])
    return r

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
