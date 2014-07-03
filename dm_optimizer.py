from __future__ import print_function

import scipy.optimize as sopt
import scipy
import numpy as np
from numpy.linalg import norm
from copy import copy

from sortedcontainers import sortedlist as sl
from functools import total_ordering

import sys

@total_ordering
class value_box:
    """ A silly class that represents points (y, x) compared on the basis of their y-components only. """
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
    """ A global optimizer using the difference map-based algorithm with or without constraints.
        Attributes:
            max_iterations      -- the maximum number of iterations to perform before aborting minimization.
            tolerance           -- the threshold above which we consider two points to be distinct.
            verbosity           -- controls debug output; higher numbers = more messages.
            first_target_ratio  -- determines how low the first target should be.
            greediness          -- how quickly should the target value be lowered.
            fun                 -- the objective function.
            pseudo              -- special value to use to determine the step size when it would otherwise be very large.
            refresh_rate        -- how frequently to update the target value, in iterations per update.
            nfev                -- total number of function evaluations.
            vals                -- the local minima collected along the way.
            lpos                -- the path the optimizer has followed along the objective function.
            minimizer_kwargs    -- keyword arguments passed to the local minimizer.

        Notes:
            vals is stored as a sorted list, since we frequently just need to get the n smallest values.
            It is a list of tuples (y, x), which is convenient since Python tuples are totally ordered according
            to their first element.
            """

    def __init__(self, fun, max_iterations=2500, target=None, constraints=[], nonsatisfaction_penalty=0, tolerance=0.000001, verbosity=0, pseudo=0.001,
                 greediness=0.05, first_target_ratio=0.9, refresh_rate=10, logfile=sys.stderr, minimizer_kwargs={}):
        self.max_iterations             = max_iterations
        self.tolerance                  = tolerance
        self.verbosity                  = verbosity
        self.first_target_ratio         = first_target_ratio
        self.greediness                 = greediness
        self.pseudo                     = pseudo
        self.constraints                = constraints
        self.nonsatisfaction_penalty    = nonsatisfaction_penalty
        self.refresh_rate               = refresh_rate
        self.logfile                    = logfile
        self.minimizer_kwargs           = minimizer_kwargs
        self._fun                       = fun # store the objective function in a 'hidden' attribute to avoid accidentally calling the unwrapped version

        if not (target is None):
            self.fixed_target = True
            self.target       = target
        else:
            self.fixed_target = False


    def logmsg(self, priority, *args, **kwargs):
        """ Forward all arguments to print if the given priority is less than or equal to the verbosity of the optimizer. The file keyword argument
            of print is set to sys.stderr.

            Arguments:
                priority -- if this value is less than `verbosity`, all other arguments are forwarded to print.

            Returns:
                Nothing.
            """
        if priority <= self.verbosity:
            print(*args, file=self.logfile, **kwargs)

    def fun(self, x):
        """ Evaluate the objective function at the given point and increment the running count of function calls. """
        self.nfev += 1
        return self._fun(x)

    def evalf(self, x):
        """ Evaluate the objective function at the given point and apply the nonsatisfaction penalty times the number of
            constraints not satisfied. """
        return self.fun(x) + sum([self.nonsatisfaction_penalty for constraint in self.constraints if not constraint(x)])

    def local_min(self, x):
        """ Find the local minimum nearest to the given point. """
        #return fmin(self.fun, x, xtol=self.tolerance, maxiter=2000, maxfun=2000, disp=self.verbosity==2)
        res = sopt.minimize(self.fun, x, **self.minimizer_kwargs)
        # it is unnecessary to manually increase self.nfev since self.fun already does this for us.
        # still, we must manually increase self.njev or self.nhev since minimize might use the Jacobian or Hessian
        if 'njev' in res.keys():
            self.njev += res.njev
        if 'nhev' in res.keys():
            self.nhev += res.nhev
        return res.x

    def best_minimum_x(self, pmin):
        """ Find the minimum in the list of past minima that is closest to the given value.

            Arguments:
                pmin -- the point from which the distances are to be calculated.

            Returns:
                _(y, x)_ -- the minimum closest to the given point.

            Notes:
                To avoid accidental rapid convergeance to a fixed point, this function will not return the true closest minimum
                if it is considered to be the same as the given point; it will instead return the second-closest minimum. Two points
                are considered the same if their distance is less than `tolerance`.
            """
        minima = sorted([x for x in self.vals
                                 if norm(x.x - pmin) >= self.tolerance
                                 #and (self.evalf(pmin) - x.y)**2 >= 0.1
                        ], key=lambda v: norm(pmin - v.x))

        if len(minima) == 0:
            raise Exception("There are no local minima far away enough from the current minimum to be considered distinct.")

        return minima[0]

    def new_target(self, best):
        """ Calculate a new target value.

            Arguments:
                best -- the current function-value.

            Returns:
                Nothing.

            Notes:
                Sets the objects internal `target` attribute to the new target value. This method should be called when the actual function value
                falls below the current target value.
            """
        # get the minimum y value recorded in our list of minima, and push down from there.
        oldtarget = self.target
        self.target = best + self.greediness * (best - min([v.y for v in self.vals if v.y - best > self.tolerance]))
        if self.target != oldtarget:
            self.logmsg(1, "Target updated to", self.target)

    def refresh_target(self):
        """ Refresh the target value.

            Arguments:
                None.

            Returns:
                Nothing.

            Notes:
                Uses the internally-maintained list of past minima to set `target` to the new target value.
            """
        if self.fixed_target:
            pass
            #self.logmsg(2, "Target is fixed. Refusing to refresh target.")
        else:
            # get the two best minima so far found
            bests = self.vals[:2] # remember, vals is sorted in ascending order of the y-value.
            # bests :: [(y,x)], too !
            #self.logmsg(1, "Two best minima so far: ", bests[0].unbox(), " and ", bests[1].unbox())
            diff = bests[0].y - bests[1].y
            if diff > 0:
                raise Exception("The difference between the two best minima should be negative!")
            #self.logmsg(2, "Y-Difference between the minima: ", diff)
            self.target = bests[0].y + self.greediness * diff
            #self.logmsg(1, "Refreshed target to", self.target)

    def make_step(self, (fv, pmin), (ynear, xnear), deltay_curr):
        if fv < ynear: #if the current local minimum is *better* than any previous one, we move *away* from the old one
            #self.logmsg(1, "Doubling back!")
            stepdir = pmin - xnear
        else: # otherwise, we move *towards* the old one
            stepdir = xnear - pmin # the direction of the step

        deltay_prev = ynear - self.target # how far away is the best previously-found local minimum from the target
        # if things are going well, then deltay_curr should be less than deltay_prev
        #self.logmsg(2, "Bes+ minimum delta:", deltay_prev)

        if (deltay_curr - deltay_prev)**2 < self.pseudo**2: # calculate the step scale.
            #self.logmsg(1, "Very close deltas.")
            stepscale = deltay_curr / self.pseudo # Use pseudo to avoid jumping away to infinity, if necessary.
        else:
            stepscale = deltay_curr / (deltay_curr - deltay_prev)

        #if abs(stepscale) < self.tolerance:
        #    stepscale = stepscale / abs(stepscale) * 2.0

        step = stepscale * stepdir

        if stepscale == 0:
            raise Exception()

        #self.logmsg(1, "Current delta y:", deltay_curr)
        #self.logmsg(1, "Stepscale:", stepscale)
        #self.logmsg(1, "Step:", step)
        #self.logmsg(1, "Distance to minimum:", norm(stepdir))

        return step

    def minimize(self, x0, x1):
        """ Perform the minimization starting given two initial points.

            Arguments:
                x0, x1 -- the initial points.

            Returns:
                _lpos_ -- the points the optimizer travelled to, the last of which is the global minimum or
                          the best minimum in the exhaustion.

            Notes:
                If the number of iterations exceeds the value set in the constructor, then the best local minimum found so far is returned.
            """
        # Ensure that our lists are empty.
        self.vals                       = sl.SortedList()
        self.lpos                       = []

        # and that our counters are zero
        self.nfev       = 0
        self.njev       = 0
        self.nhev       = 0
        self.iteration  = 0

        # Get the first minimum
        x0min = self.local_min(x0)
        f0 = self.evalf(x0min)

        self.vals.add(value_box( (f0, x0min) ))
        self.target = self.first_target_ratio * f0

        nx1 = x1
        self.lpos = [(self.evalf(nx1), nx1)]

        #self.logmsg(2, "Entering loop.")

        try:
            for self.iteration in xrange(self.max_iterations):
                #self.logmsg(2, "Guess point for this iteration:", nx1)
                pmin = self.local_min(nx1)
                fv = self.evalf(pmin) # the function value at the local minimum for this iteration
                deltay_curr = fv - self.target # how far away is our current minimum from the target value

                #self.logmsg(1, "Got delta:", deltay_curr)

                if deltay_curr**2 <= self.tolerance**2: # we've fallen below the target, so we create a new target
                    #self.logmsg(1, "Beat target!")
                    self.new_target(fv) # to get the next target, all we need is the current function value and the list of minima so far
                    deltay_curr = fv - self.target # recalculate delta for the new target
                    #self.logmsg(1, "New delta:", deltay_curr)
                else:
                    pass
                    #self.logmsg(1, "Target y-value", self.target)

                if deltay_curr == 0:
                    raise Exception()

                # let's take the *best* minimum, i.e. the minimum closest to our current minimum
                ynear, xnear = self.best_minimum_x(pmin).unbox()

                #self.logmsg(3, "Nearest old minimum f", xnear, " = ", ynear, sep='')

                step = self.make_step( (fv, pmin), (ynear, xnear), deltay_curr)

                nx1 += step # actually take the step
                self.lpos.append((self.evalf(nx1), copy(nx1))) # add the new position to the list of past positions
                #self.logmsg(2, "Took step ", step)

                self.vals.add(value_box( (fv, copy(pmin)) ))

                if norm(step) < self.tolerance:
                    #self.logmsg(1, "Found fixed point: f", pmin, " = ", ynear, sep='')
                    #self.logmsg(1, "Step was:", step)
                    self.new_target(fv)

                    #self.lpos.append( (fv, copy(pmin) ) )
                    #res = scipy.optimize.OptimizeResult()
                    #res.nit = self.iteration
                    #res.success = True
                    #res.message  = ["Fixed-point found."]
                    #res.status   = 0
                    #res.x = pmin
                    #res.fun = fv
                    #res.njev = self.njev
                    #res.nfev = self.nfev
                    #return res

                if (not self.fixed_target) and self.iteration % self.refresh_rate == 0: # every `refresh_rate` iterations, we refresh the target value, to avoid stale convergeance
                    oldtarget = self.target
                    self.refresh_target() # mutates `target`
        except:
            #self.logmsg(-1, "An error occurred while optimizing:")
            raise

        # we reach this point if no exceptions have been raised and max_iterations has been exhausted
        self.lpos.append(min(self.vals, key=lambda v: v.y).unbox())
        fv, pmin = self.lpos[-1]

        res = sopt.OptimizeResult()
        res.nit     = self.iteration
        res.success = True
        res.message = ["Maximum number of iterations reached."]
        res.status  = 1
        res.x       = pmin
        res.fun     = fv
        res.njev    = self.njev
        res.nfev    = self.nfev
        res.lpos    = self.lpos
        return res

def minimize(f, x1, x2, **kwargs):
    """ Construct a dm_optimizer object and run its minimize method with the given initial points.
        The result is packaged into a scipy OptimizeResult
        object.

        Arguments:
            f       -- the objective function.
            x1, x2  -- the initial points.
            kwargs  -- passed to the constructor of dm_optimizer

        Returns:
            A scipy.optimize.OptimizeResult object populated with typical data.
            Refer to the scipy documentation on OptimizeResult for more information.
        """
    optimizer = dm_optimizer(f, **kwargs)

    #try:
    return optimizer.minimize(x1, x2)
    #except Exception as e:
    #    res = sopt.OptimizeResult()
    #    res.message = str(e)
    #    res.status  = 2
    #    res.success = False
    #    res.optimizer = optimizer # we'll the broken optimizer in here in case we need it to recover some failure information
    #    return res

