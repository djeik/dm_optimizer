#!/usr/bin/env python

""" A nasty hack for using Mathematica's local minimizer in Python by invoking
    a MathematicaScript subprocess. The function to minimize is baked into the
    script MathematicaFindMinimum.m, so change it there in order to minimize
    difference function.
"""

import subprocess as sp
import json

def mathematica_findminimum(**kwargs):
    """ The keyword arguments are forwarded as-is to the mathematica script as
        JSON. The output of the mathematica script is deserialized from JSON
        and return as-is as well.
        Note: this is EXTREMELY SLOW.
    """
    math = sp.Popen(["./MathematicaFindMinimum.m"], stdin=sp.PIPE, stdout=sp.PIPE)
    out, err = math.communicate(json.dumps(kwargs))
    return json.loads(out)
