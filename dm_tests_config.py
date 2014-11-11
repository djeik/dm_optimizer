#!/usr/bin/env python2

import dm_utils as dmu
from numpy.linalg import norm

mp_subproc_count = 4

poll_names = ["function_value", "best_minimum", "step_size"] # the names of the things extracted from the optimizer internal state

sampler_defaults = {"dimensions":5, "range":(-100, 100)}
experiment_defaults = {"runs":100, "success_threshold":0.001, "terminate_on_optimum":False}
dm_defaults = {"max_iterations":250, "stepscale_constant":0.1,
        "tolerance":0.000001, "minimum_distance_ratio":3.0}
sa_defaults = {"niter":250}
iterations_config = {"end":1000}

plots_config = {"individual_color":"0.6", "average_color":"blue"}

contour_resolution = 1.0 / 8 # units per sample

def get_sample_count(units):
    """ For a given number of units, calculate the number of samples that
        should be taken along an interval.  This function is used particularly
        when plotting the countours of an objective function. These functions
        are extremely bumpy, so high resolutions are needed to make good plots.
        """
    return units / contour_resolution

def get_range_sizes(test):
    """ For a given test, get a list of the sizes of each of its variables' range.
        """
    if test["range"] is None:
        return list(repeat(
            sampler_defaults["range"],
            test["dimensions"] or test_functions.SAMPLER_DEFAULTS["dimensions"]))
    else:
        return [r[1] - r[2] for r in test["range"]]

class solver_callback(object):
    def __init__(self, optimum=float("nan"), experiment_settings=experiment_defaults):
        self.vs = []
        self.optimum = optimum
        self.experiment_settings = experiment_settings

class dm_callback(solver_callback):
    def __init__(self, *args, **kwargs):
        super(dm_callback, self).__init__(*args, **kwargs)

    def __call__(self, solver):
        self.vs.append( (
            solver.current_minimum.y,
            solver.vals[0].y,
            norm(solver.step) ) )

        if self.experiment_settings["terminate_on_optimum"]:
            return (solver.vals[0].y - self.optimum)**2 <= \
                self.experiment_settings["success_threshold"]**2

class sa_callback(solver_callback):
    def __init__(self, *args, **kwargs):
        super(sa_callback, self).__init__(*args, **kwargs)

    def __call__(self, x, f, accept):
        if accept:
            self.vs.append( (f, x) )

        if self.experiment_settings["terminate_on_optimum"]:
            return (f - self.optimum)**2 <= \
                    self.experiment_settings["success_threshold"]**2

def make_dm_defaults(optimum=float("nan")):
    """ Construct a configuration object for the DM solver, whose callback will
        optionally be generated with a given optimum.
        """
    defaults = dict(dm_defaults)
    defaults["callback"] = dm_callback(optimum)
    return defaults

def make_sa_defaults(optimum=float("nan")):
    defaults = dict(sa_defaults)
    defaults["callback"] = sa_callback(optimum)
    return defaults

optimizers = {"dm":{"tag":"dm", "optimizer":dmu.randomr_dm, "config_gen":make_dm_defaults},
              "sa":{"tag":"sa", "optimizer":dmu.randomr_sa, "config_gen":make_sa_defaults}}

def optimizer_config_gen(optimizer, optimum=float("nan"), extra_config={}):
    """ Construct a configuration object for the given optimizer. The optimizer
        should be a value from the ``optimizers'' dictionary in this module.
        """
    optimizer["config"] = optimizer["config_gen"](optimum)
    for (k, v) in extra_config.items():
        optimizer["config"][k] = v
    return optimizer
