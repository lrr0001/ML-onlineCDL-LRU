#!/bin/bash
databasename='BSDS500/'
for ii in {0..48..4}
  do
    for logrho in {-6..9..3}
      do
        python CSCadmmScript.py $databasename $logrho $ii
      done
    for loglpstz in {0..15..3}
      do
        python CSCfistaScript.py $databasename $loglpstz $ii
      done
  done
