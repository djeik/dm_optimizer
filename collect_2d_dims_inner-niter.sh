#!/bin/bash

# Data collection script. The result set is a two-dimensional matrix stored in
# the filesystem, (dimensions, inner iterations). The inner iteration count is
# the maximum number of iterations to use in the local optimization procedure
# of DM, which occurs DM_NITER + 1 times.
#
# If $p is the path to results and $N is a number, then $p/$N is the directory
# in which the data for the runs on dimension $N are stored.
# The naming scheme for the files stored in $p/$N is as follows:
#     $S-$M.json
# where $S is the short name of the solver ("dm" or "sa") and $M is the number
# of inner iterations. Since the inner iterations affects only DM, we run the
# simulated annealing run only once, and give it a nominal value of $M "x".
# So we have files looking like dm-10.json, dm-150.json, sa-x.json, etc.
#
# This script will create a subdirectory of the main results directory named
# after the current date in ISO-8601 format and prefixed with "mma-".

set -e

now="$(date -Iseconds)"
resultdir="../results/mma-$now"

mkdir "$resultdir"

touch "$resultdir/dimensions"

for dim in {2..15} ; do
    export DM_DIMENSIONS=$dim
    export DM_NITER=100
    export DM_TOLERANCE="1e-8"
    export DM_RUNCOUNT=35
    export DM_RANGE="-1000,1000"
    export DM_STARTDISTANCE="5.0"

    mkdir "$resultdir/$dim"

    echo "SA dim $dim" >&2

    # Simulated annealing doesn't care about the DM_INNERNITER parameter, so we
    # just record it separately, for each dimension.
    DM_SOLVERS="sa" ./DMData.m > "$resultdir/$dim/sa-x.json"

    for innerniter in {10,25,50,100,250,500} ; do
        echo "DM dim $dim / innerniter $innerniter" >&2

        outpath="$resultdir/$dim/dm-${innerniter}.json"
        DM_SOLVERS="dm" DM_INNERNITER=$innerniter ./DMData.m > "$outpath"
    done
done
