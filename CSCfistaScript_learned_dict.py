import ML_FISTA_eval_jpeg_artifact_removal as fista
import sys

if __name__ == '__main__':
    databasename = sys.argv[1]
    lpstz = 2**(int(sys.argv[2])/2)
    noi = int(sys.argv[3])
    pklfile = sys.argv[4]
    #lpstz = 100
    spe = 16
    noe = 64
    fista.test_FISTA_CSC_saved_dict(lpstz = lpstz,noi = noi,databasename = databasename,steps_per_epoch = spe, num_of_epochs = noe,pklfile = pklfile)
