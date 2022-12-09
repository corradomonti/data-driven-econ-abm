import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mlflow
import numpy as np
# import scipy.stats

import itertools
import pickle

def make_plots(M, M_est, P=None, P_est=None, Y=None, save_pickle=True):
    _T, L, K = M.shape
    
    if save_pickle:
        with open("results-for-plots.pickle", 'wb') as f:
            pickle.dump((M, M_est, P, P_est, Y), f)
        mlflow.log_artifact("results-for-plots.pickle")
            
    for matrix, title, filename in [
        (M[0], "True $M_0$", 'plt_M0_true.png'), 
        (M_est[0], "Estimated $M_0$", 'plt_M0_est.png'), 
        (M[-1], "True $M_T$", 'plt_MT_true.png'), 
        (M_est[-1], "Estimated $M_T$", 'plt_MT_est.png'), 
    ]:
        plt.imshow(matrix)
        plt.xlabel("Income class")
        plt.ylabel("Location")
        plt.title(title)
        plt.colorbar()
        plt.savefig(filename)
        mlflow.log_artifact(filename)
        plt.clf()
    
    ymax = max(np.max(M), np.max(M_est)) + 10
    _fig, axs = plt.subplots(L, K, figsize=(L * 4, K * 6))
    for l in range(L):
        for k in range(K):
            axs[l, k].set_title(f"Location {l}, class {k}")
            axs[l, k].plot(M[:, l, k], label="Original")
            axs[l, k].plot(M_est[:, l, k], label="Estimate")
            axs[l, k].set_ylim(0, ymax)
            axs[l, k].set_ylabel("Population")
            axs[l, k].grid()
            axs[l, k].legend()
    plt.savefig("population-class.png")
    mlflow.log_artifact("population-class.png")
    plt.clf()
    
    for matrix, name in [(M, 'true'), (M_est, 'est')]:
        plt.figure(figsize=(10, 5))
        plt.title("Num. persons in each class (original)")
        plt.plot(np.einsum('tlk->tk', matrix))
        plt.ylim(0)
        plt.ylabel("Population")
        plt.xlabel("Time")
        plt.grid()
        # plt.legend(handles=[
        #     mpatches.Patch(label=f"Income: {int(y)}", **color)
        #     for y, color in zip(Y, plt.rcParams["axes.prop_cycle"])
        # ])
        labels = ["Poor", "Middle", "Rich"] if K == 3 else [f"Class {k}" for k in range(K)]
        plt.legend(handles=[
            mpatches.Patch(label=label, **color)
            for label, color in zip(labels, plt.rcParams["axes.prop_cycle"])
        ])
        plt.savefig(f"population-total-{name}.png")
        mlflow.log_artifact(f"population-total-{name}.png")
        plt.clf()
        
    plt.figure(figsize=(10, 5))
    plt.title("Total number of agents")
    plt.plot(np.einsum('tlk->t', M), label="Original")
    plt.plot(np.einsum('tlk->t', M_est), label="Estimate")
    plt.ylim(0)
    plt.ylabel("Population")
    plt.xlabel("Time")
    plt.grid()
    plt.legend()
    plt.savefig("population-total.png")
    mlflow.log_artifact("population-total.png")
    plt.clf()
    
    # if Y is not None:
    #     plt.figure(figsize=(10, 5))
    #     plt.plot([scipy.stats.entropy(np.einsum('k,lk->l', Y, Mt), base=2) for Mt in M])
    #     plt.ylim(0)
    #     plt.ylabel("Income entropy")
    #     plt.xlabel("Time")
    #     plt.grid()
    #     plt.legend()
    #     plt.savefig("income-entropy.png")
    #     mlflow.log_artifact("income-entropy.png")
    #     plt.clf()
    
    for label, m_tensor in [("original", M), ("estimated", M_est)]:
        plt.figure(figsize=(10, 4))
        for k, color in zip(range(m_tensor.shape[2]), 
                            itertools.chain(iter("ryg"), itertools.cycle("k"))):
            plt.plot(m_tensor[:, :, k], '-', color=color)
        plt.ylabel("Population")
        plt.xlabel("Time")
        plt.grid()
        plt.savefig(f"population-{label}.png")
        mlflow.log_artifact(f"population-{label}.png")
        plt.clf()
        
    for label, prices in [("original", P), ("estimated", P_est)]:
        if prices is not None:
            plt.figure(figsize=(10, 4))
            plt.plot(prices, 'o-')
            plt.ylim((0, None))
            plt.ylabel("Price")
            plt.xlabel("Time")
            plt.title(title.title())
            plt.grid()
            plt.savefig(f"prices-{label}.png")
            mlflow.log_artifact(f"prices-{label}.png")
            plt.clf()
