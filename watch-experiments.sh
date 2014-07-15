#!/bin/bash

while true 
do 
    find experiments -name "averages.txt" -printf "%T@ %p\n" \
    | sort -n | tail -1 | cut -f2- -d" " | xargs -d\\n cat | column -ts , \
    | less 
    sleep 1 
done
