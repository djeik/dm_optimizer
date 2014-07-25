#!/usr/bin/env python2

from dm_utils import *

def dm_poll_callback(self):
    self.vs.append( (self.fv, self.vals[0].y, norm(self.step)) )

minimization = 1
maximization = -1

tests = map(lambda xs: dict(zip(
        ["name",        "function",                        "optimization_type", "dimensions", "range",      "optimum"], xs)),
        [("ackley",     "unwrap_bench(bench.ackley)",      minimization,        None,         (-15, 30),     0),
        ("cigar",       "unwrap_bench(bench.cigar)",       minimization,        None,         None,          0),
        ("sphere",      "unwrap_bench(bench.sphere)",      minimization,        None,         None,          0),
        ("bohachevsky", "unwrap_bench(bench.bohachevsky)", minimization,        None,         (-100, 100),   0),
        ("griewank",    "unwrap_bench(bench.griewank)",    minimization,        None,         (-600, 600),   0),
        ("h1",          "unwrap_bench(bench.h1)",          maximization,        2,            (-100, 100),   2),
        ("himmelblau",  "unwrap_bench(bench.himmelblau)",  minimization,        2,            (-6, 6),       0),
        ("rastrigin",   "unwrap_bench(bench.rastrigin)",   minimization,        None,         (-5.12, 5.12), 0),
        ("rosenbrock",  "unwrap_bench(bench.rosenbrock)",  minimization,        None,         None,          0),
        ("schaffer",    "unwrap_bench(bench.schaffer)",    minimization,        None,         (-100, 100),   0),
        ("schwefel",    "unwrap_bench(bench.schwefel)",    minimization,        None,         (-500, 500),   0),
        ("simon_f2",    "simon_f2",                        minimization,        2,            (-100, 100),   0)])
        ## The following functions are 'weird' in some way that makes testing too difficult.
        #(unwrap_bench(bench.rastrigin_scaled),
        #(bench.rastrigin_skew
        #(unwrap_bench(bench.rand), None, None, None, None),
        #(unwrap_bench(bench.plane), minimization, None, None, 0),

poll_names = ["function_value", "best_minimum", "step_size"] # the names of the things extracted from the optimizer internal state

sampler_defaults = {"dimensions":5, "range":(-100, 100)}
experiment_defaults = {"runs":250, "success_threshold":0.001}
dm_defaults = {"refresh_rate":4, "max_iterations":250, "callback":dm_poll_callback}
sa_defaults = {"niter":250}

optimizers = {"dm":{"tag":"dm", "optimizer":randomr_dm, "config":dm_defaults},
              "sa":{"tag":"sa", "optimizer":randomr_sa, "config":sa_defaults}}
