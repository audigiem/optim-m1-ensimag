#!/usr/bin/sh
cd src
ls *.py | while IFS= read -r fname; do mv $fname $fname.corr; done
ls *.py.bkp | while IFS= read -r fname; do mv $fname $(echo $fname | sed 's/\(.*\).py.bkp/\1.py/g'); done
cd ..
