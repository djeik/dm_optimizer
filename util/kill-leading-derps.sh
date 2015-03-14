#!/bin/bash

# When I ran the DMData.m initially, I forgot to remove a Print statement in it
# which output the name of the solvers to run the test on, so these found them-
# selves prefixed to the output file in the form "{dm}" or "{sa}"

# This script will remove such a first line from all files found in the given
# tree whose first line is of the form "{SomeAlphaNumericCharacters}".

find "$1" -type f | while read line
do
    if (head -n 1 "$line" | grep '{[[:alnum:]]\+}') ; then
        sed -i 1d "$line"
    fi
done
