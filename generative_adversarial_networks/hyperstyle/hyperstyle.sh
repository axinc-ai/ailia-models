#!/usr/bin/env bash

if command -v python3 > /dev/null 2>&1; then
    py_cmd () {
        python3 "$@"
    }
elif command -v python > /dev/null 2>&1; then
    py_cmd () {
        python "$@"
    }
else
    "python3/python command not found"
    exit 1
fi

logfile="hyperstyle.log"
for i in {0..2}; do
    echo "Run $i" >> "$logfile"
    { time py_cmd hyperstyle.py --adaptation --use_dlib; } >> "$logfile" 2>&1
    echo "" >> "$logfile"
done
