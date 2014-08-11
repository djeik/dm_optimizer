#!/usr/bin/env python2

import dm_utils as dmu
from numpy.linalg import norm

def dm_poll_callback(self):
    self.vs.append( (self.fv, self.vals[0].y, norm(self.step)) )

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

optimizers = {"dm":{"tag":"dm", "optimizer":dmu.randomr_dm, "config":dm_defaults},
              "sa":{"tag":"sa", "optimizer":dmu.randomr_sa, "config":sa_defaults}}

iterations_config = {"start":25, "end":1000, "step":25}
