#!/usr/bin/env python2

import dm_utils as dmu
from numpy.linalg import norm

minimization = 1
maximization = -1

tests = map(lambda xs: dict(zip(
        ["name",        "function",                        "optimization_type", "dimensions", "range",      "optimum"], xs)),
        [("ackley",     "unwrap_bench(bench.ackley)",      minimization,        9,            (-15, 30),     0),  # best
        ("bohachevsky", "unwrap_bench(bench.bohachevsky)", minimization,        7,            (-100, 100),   0),  # best
        ("cigar",       "unwrap_bench(bench.cigar)",       minimization,        10,           None,          0),  # large
        ("griewank",    "unwrap_bench(bench.griewank)",    minimization,        2,            (-600, 600),   0),  # best
        ("h1",          "unwrap_bench(bench.h1)",          maximization,        4,            (-100, 100),   2),  # required
        ("himmelblau",  "unwrap_bench(bench.himmelblau)",  minimization,        4,            (-6, 6),       0),  # required
        ("rastrigin",   "unwrap_bench(bench.rastrigin)",   minimization,        6,            (-5.12, 5.12), 0),  # best
        ("rosenbrock",  "unwrap_bench(bench.rosenbrock)",  minimization,        7,            None,          0),  # large, but slow
        ("schaffer",    "unwrap_bench(bench.schaffer)",    minimization,        4,            (-100, 100),   0),  # best
        ("schwefel",    "unwrap_bench(bench.schwefel)",    minimization,        4,            (-500, 500),   0),  # best
        ("simonf2",     "simonf2",                        minimization,        2,            (-100, 100),   0),  # required
        ("sphere",      "unwrap_bench(bench.sphere)",      minimization,        10,           None,          0)]) # large

poll_names = ["function_value", "best_minimum", "step_size"] # the names of the things extracted from the optimizer internal state

sampler_defaults = {"dimensions":5, "range":(-100, 100)}
experiment_defaults = {"runs":250, "success_threshold":0.001}
dm_defaults = {"max_iterations":100}
sa_defaults = {"niter":100}

plots_config = {"individual_color":"0.6", "average_color":"blue"}

class solver_callback(object):
    def __init__(self, optimum=float("nan"), experiment_settings=experiment_defaults):
        self.vs = []
        self.optimum = optimum
        self.experiment_settings = experiment_defaults

class dm_callback(solver_callback):
    def __init__(self, *args, **kwargs):
        super(dm_callback, self).__init__(*args, **kwargs)

    def __call__(self, solver):
        self.vs.append( (solver.fv, solver.vals[0].y, norm(solver.step)) )
        if (solver.vals[0].y - self.optimum)**2 <= self.experiment_settings["success_threshold"]**2:
            return True

class sa_callback(solver_callback):
    def __init__(self, *args, **kwargs):
        super(sa_callback, self).__init__(*args, **kwargs)

    def __call__(self, x, f, accept):
        if accept:
            self.vs.append( (f, x) )

        if (f - self.optimum)**2 <= self.experiment_settings["success_threshold"]**2:
            return True


def make_dm_defaults(optimum=float("nan")):
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
    optimizer["config"] = optimizer["config_gen"](optimum)
    for (k, v) in extra_config.items():
        optimizer["config"][k] = v
    return optimizer

iterations_config = {"start":25, "end":1000, "step":25}

solved_vs_iterations_subproc_count = 4
