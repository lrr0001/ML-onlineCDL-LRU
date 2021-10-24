#!/bin/bash
databasename='BSDS500/'

modelname + 'checkpoint_epoch_{epoch:02d}.ckpt'
modelname='ML_LRA_'
checkpointstr='checkpoint_epoch_'
extension='.ckpt.pkl'
for ii in {1..9..1}
  do
    python CSCadmmScript_learned_dict.py $databasename 0 16 $modelname$checkpointstr$ii$extension
  done
modelname='ML_FISTA_'
extension='.pkl'
for ii in {1..9..1}
  do
    python CSCfistaScript_learned_dict.py $databasename 12 48 $modelname$checkpointstr$ii$extension
  done

