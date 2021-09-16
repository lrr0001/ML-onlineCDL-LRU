#!/bin/bash
databasename='simpleTest/'
for ii in {0..48..4}
  do
    for logrho in {-6..3..3}
      do
        python CSCadmmScript.py $databasename $logrho $ii
      done
    for loglpstz in {7..10..1}
      do
        python CSCfistaScript.py $databasename $loglpstz $ii
      done
  done
