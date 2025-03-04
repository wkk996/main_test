import numpy as np
from nn_mpc_cbf import MPCNN, loop_test
import util

def main():
    np.random.seed()
    controllerNN = MPCNN()
    loop_test(controllerNN, N=1000)
if __name__ == '__main__':
    main()
