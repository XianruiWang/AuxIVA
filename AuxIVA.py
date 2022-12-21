# Author: Xianrui Wang, CIAIC, NWPU & LMS, FAU
# Contact: wangxianrui@mail.nwpu.edu.cn
# Date: 21-04-2022
# All copyrights reserved

from re import L, X
from tkinter import W
from traceback import print_last
from basetool import *



def AuxIVA(X_FTM=None, n_src=None, n_iter=20, distribution="laplace", return_filter = False):
    """
    AuxIVA
    ======

    Blind Source Separation using independent vector analysis based on auxiliary function.
    This function will separate the input signal into statistically independent sources
    without using any prior information.

    Parameters
    -------
    X_FTM: multichannel mixed data in frequency domain
    n_src: number of sources
    n_iter: number of iterations
    distribution: gauss or laplace

    Return
    -------
    YHat_FTN: estimated signal n_freq * n_frames *

    References
    ----------
    .. [1] N. Ono, *Stable and fast update rules for independent vector analysis based
        on auxiliary function technique,* Proc. IEEE, WASPAA, pp. 189-192, Oct. 2011.

    ======
    """

    eps = 1e-8
    threshold = 1e-6
    n_freq,  n_frame, n_channel = X_FTM.shape

    # parameters check
    assert (n_src <= n_channel), "The sources cannot be more than microphones"
    if distribution not in ["laplace", "gauss"]:
        raise ValueError("distribution not included, please use laplace or gaussian")

    # if M>N, Principle Component Analysis (PCA) should be implemented
    X_FTN = pca(X_FTM, n_src)
    X_FNT = X_FTN.transpose([0, 2, 1]).copy()  # size of F*N*T
    

    # memory allocation
    # YHat_FNT = np.zeros([n_freq, n_sr, n_frame])
    eyes = np.tile(np.eye(n_src, n_src), (n_freq, 1, 1))
    W_FNN = np.tile(np.eye(n_src, n_src, dtype=X_FTN.dtype), (n_freq, 1, 1))
    loss_old = np.inf

    YHat_FNT = np.zeros((n_freq, n_src, n_frame), dtype=X_FTM.dtype)
    def separate():
        """
        separate signal
        """
        YHat_FNT[:, :, :] = W_FNN @ X_FNT

    def cal_loss():
        """
        calculate cost function
        """
        G_NT = np.sqrt(np.sum(np.abs(YHat_FNT)**2, axis=0))
        G_N = np.mean(G_NT, axis=1)
        G = np.sum(G_N)
        lod_det = np.sum(np.log(np.abs(np.linalg.det(W_FNN))))
        loss = (G-lod_det)/(n_freq*n_src)
        loss_imp = np.abs(loss-loss_old)/abs(loss)
        loss_last = loss
        return loss, loss_last, loss_imp

    def cal_r_inv():
        """
        calculate r in eq. 34
        """
        if distribution == "laplace":
            r = np.linalg.norm(YHat_FNT, axis=0)
        elif distribution == "gauss":
            r = (np.linalg.norm(YHat_FNT, axis=0)**2)/n_freq
        r[r < eps] = eps
        r_inv = 1.0/r
        return r_inv

    # estimate demix matrix with n_iter epoches
    for epoch in range(n_iter):
        separate()
        if epoch % 10 == 0:
            loss_current, loss_old, loss_imporve = cal_loss()
            print("loss has been improved by {} at lastest 10 iteration".format(loss_imporve))
            if loss_imporve <= threshold:
                break
        r_inv_NT = cal_r_inv()
        for n in range(n_src):
            V_FNN = np.matmul(X_FNT*r_inv_NT[None, n, None, :],
                              np.conj(X_FNT.swapaxes(1, 2)))/n_frame
            WV_FNN = np.matmul(W_FNN, V_FNN)
            u = np.linalg.solve(WV_FNN, eyes[:, :, n])  # eq. 36
            u_temp1 = np.sqrt(np.matmul(np.matmul(np.conj(u[:, None, :]), V_FNN),
                                        u[:, :, None]))
            u_temp2 = (u[:, :, None]/u_temp1)[:, :, 0]
            W_FNN[:, n, :] = np.conj(u_temp2)
    separate()
    YHat_TFN = YHat_FNT.transpose([2, 0, 1])
    spec_FT = X_FTM[:,:,0]
    X_TF = spec_FT.T
    Z = projection_back(YHat_TFN, X_TF)
    YHat_TFN *= np.conj(Z[None, :, :])
    YHat_FTN = YHat_TFN.transpose([1, 0, 2])
    if return_filter:
        return YHat_FTN, W_FNN
    else:
        return YHat_FTN