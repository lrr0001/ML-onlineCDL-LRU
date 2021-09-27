import ML_ADMM_eval_jpeg_artifact_removal as admm
import sys

if __name__ == '__main__':
    databasename = sys.argv[1]
    rho = 2**int(sys.argv[2])
    noi = int(sys.argv[3])
    pklfile = sys.argv[4]
    alpha = 1.5
    spe = 16
    noe = 64
    admm.test_ADMM_CSC_saved_dict(rho = rho,alpha_init=alpha,noi = noi,databasename = databasename,steps_per_epoch = spe, num_of_epochs = noe,pklfile = pklfile)
