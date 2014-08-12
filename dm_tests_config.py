#!/usr/bin/env python2

import dm_utils as dmu
from numpy.linalg import norm

minimization = 1
maximization = -1

tests = map(lambda xs: dict(zip(
        ["name",        "function",                        "optimization_type", "dimensions", "range",      "optimum"], xs)),
        [("ackley",     "unwrap_bench(bench.ackley)",      minimization,        9,            (-15, 30),     0),  # best
        ("bohachevsky", "unwrap_bench(bench.bohachevsky)", minimization,        11,           (-100, 100),   0),  # best
        ("cigar",       "unwrap_bench(bench.cigar)",       minimization,        10,           None,          0),  # large
        ("griewank",    "unwrap_bench(bench.griewank)",    minimization,        1,            (-600, 600),   0),  # best
        ("h1",          "unwrap_bench(bench.h1)",          maximization,        2,            (-100, 100),   2),  # required
        ("himmelblau",  "unwrap_bench(bench.himmelblau)",  minimization,        2,            (-6, 6),       0),  # required
        ("rastrigin",   "unwrap_bench(bench.rastrigin)",   minimization,        6,            (-5.12, 5.12), 0),  # best
        ("rosenbrock",  "unwrap_bench(bench.rosenbrock)",  minimization,        7,            None,          0),  # large, but slow
        ("schaffer",    "unwrap_bench(bench.schaffer)",    minimization,        2,            (-100, 100),   0),  # best
        ("schwefel",    "unwrap_bench(bench.schwefel)",    minimization,        2,            (-500, 500),   0),  # best
        ("simon_f2",    "simon_f2",                        minimization,        2,            (-100, 100),   0),  # required
        ("sphere",      "unwrap_bench(bench.sphere)",      minimization,        10,           None,          0)]) # large

poll_names = ["function_value", "best_minimum", "step_size"] # the names of the things extracted from the optimizer internal state

sampler_defaults = {"dimensions":5, "range":(-100, 100)}
experiment_defaults = {"runs":75, "success_threshold":0.001}
dm_defaults = {"refresh_rate":5, "max_iterations":100, "callback":dm_poll_callback, "verbosity":"any"}
sa_defaults = {"niter":100}

class solver_callback:
    def __init__(self, optimum=float("nan"), experiment_settings=experiment_defaults):
        self.vs = []
        self.optimum = optimum
        self.experiment

class dm_callback(solver_callback):
    def __init__(*args, **kwargs):
        super(*args, **kwargs)

    def __call__(self, solver):
        self.vs.append( (self.fv, self.vals[0].y, norm(self.step)) )
        if (solver.vals[0].y - self.optimum)**2 <= self.experiment_settings["success_threshold"]**2:
            return True

class sa_callback:
    def __init__(*args, **kwargs):
        super(*args, **kwargs)

    def __call__(self, x, f, accept):
        if accept:
            self.vs.append( (f, x) )

        if (f - self.optimum)**2 <= self.experiment_settings["success_threshold"]**2:
            return True


def make_dm_defaults(optimum=float("nan")):
    defaults = dict(dm_defaults)
    defaults["callback"] = dm_callback(optimum)
    return defaults

def make_sa_callback(optimum=float("nan")):
    defaults = dict(sa_defaults)
    defaults["callback"] = sa_callback(optimum)
    return defaults

optimizers = {"dm":{"tag":"dm", "optimizer":dmu.randomr_dm, "config_gen":make_dm_defaults},
              "sa":{"tag":"sa", "optimizer":dmu.randomr_sa, "config_gen":make_sa_defaults}}

def optimizer_config_gen(optimizer, optimum=float("nan")):
    optimizer["config"] = optimizer["config_gen"](optimum)

iterations_config = {"start":25, "end":1000, "step":25}
