#!/bin/bash

cmd='find experiments -name "averages.txt" -printf "%T@ %p\n" | sort -n | tail -1 | cut -f2- -d" " | xargs -d\\n cat | column -ts ,'

watch "$cmd"
