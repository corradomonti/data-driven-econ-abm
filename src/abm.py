from utils import homophily_matrix

import numpy as np

from collections import namedtuple
import logging

EPSILON = 1E-50

class ABM:
    def __init__(self,
        # Number of locations
        L = 4,
        # Number of apartments per location
        N = 1000,
        # Number of income classes
        K = 3,
        # Number of incomers
        Q = 200,
        # Incomes
        Y = None,
        # Income distribution of new comers
        Gammak = None,
        # Vector of intrinsic attractiveness
        A0 = None,
        # Probability to put a house on sale
        alpha = 0.1,
        # Weight of attractiveness
        beta = 0.9,
        # Adjustment rate of reservation prices
        delta = 0.1,
        # # Convertion rate from number of buyers and adjustment in reservation price
        # sigma = 30,
        # mu = 0,
        # Bargaining power of sellers
        nu = 0.1,
        # Weight will to stay
        tau=2.,
        # Use homophily instead of wealth to determine attractiveness. 
        homophily=None,
        # This needs to be either None (use wealth) or an integer κ.
        # κ is how many classes around class i are considered homophiliac.
    ):
        self.L = L
        self.N = N
        self.K = K
        self.Q = Q
        self.Y = Y
        self.beta = beta
        self.alpha = alpha
        self.nu = nu
        self.delta = delta
        self.tau = tau
        self.Gammak = Gammak if Gammak is not None else (np.ones(K) / K)
        self.A0 = A0 if A0 is not None else np.ones(L)
        self.Y = Y if Y is not None else np.power(np.sqrt(10), np.arange(self.K)) * 10
        
        self.homophily = homophily
        if self.homophily is not None:
            self.H_matrix = homophily_matrix(self.K, self.homophily)
        logging.info("Generative ABM initialized with these parameters: %s", self.__dict__)

    def run_one_timestep(self, P, M, R, Y=None):
        if Y is None:
            Y = self.Y
            
        # Attractiveness
        if self.homophily is None:
            # AY_unnorm Corresponds to A^Y_{i} = (M_{i,k} * Y_k) ** tau
            AY_unnorm = np.power(
                np.einsum('ik,k->i', M/M.sum(axis=1).reshape(-1,1), Y), self.tau)
            A = self.A0 * (AY_unnorm / np.mean(AY_unnorm))
        else:
            AH_unnorm = np.power(np.einsum('xk,kj->jx', M + 1, self.H_matrix), self.tau)
            A = self.A0 * (AH_unnorm / np.mean(AH_unnorm))
        
        # Matrix of probabilities
        affordability = np.maximum(0, np.tile(Y, [self.L, 1]).T - np.tile(P, [self.K, 1]))
        Pi_unnorm = (A**self.beta * affordability**(1-self.beta)).T
        Pi = Pi_unnorm / (np.sum(Pi_unnorm, 0) + EPSILON)

        # Number of buyers of class k at location X is a matrix
        Nb = np.zeros((self.L, self.K))
        for j in range(self.K):
            if np.sum(Pi[:, j]) > 0:
                #Nb[:, j] = np.random.multinomial(self.Q*self.Gammak[j], Pi[:, j])
                #Nb[:, j] = np.floor(self.Q * self.Gammak[j] * Pi[:, j])
                Nb[:, j] = self.Q * self.Gammak[j] * Pi[:, j]
        # Number of sellers of class k at location X is also a matrix
        #Ns = np.floor(R + (self.N-R) * self.alpha)
        Ns = R + (self.N-R) * self.alpha

        
        # Sale reservation price
        bs_factor = np.sum(Nb, axis=1)/Ns
        if not np.all(np.isfinite(bs_factor)):
            raise Exception(f"Overflow in bs_factor: {bs_factor}\n"
                            f"ΣNb - Ns:\n{np.sum(Nb, axis=1)-Ns}\n")
        Ps = P*(1-self.delta + self.delta * np.tanh(bs_factor))

        # Select agents with probability depending on how much higher the price is
        prob_from_income = np.tile(Y, [self.L, 1]) - np.tile(Ps, [self.K, 1]).T
        prob_by_class_unnorm = np.sqrt(np.maximum(prob_from_income, 0)) * Nb
        prob_by_class = (prob_by_class_unnorm.T / (np.sum(prob_by_class_unnorm, 1) + EPSILON)).T

        population_distr = (M.T / np.sum(M, 1)).T

        Nd = np.minimum(np.sum(Nb, axis=1), Ns)

        Db = np.full((self.L, self.K), np.nan)
        for i in range(self.L):
            Db[i] = np.random.multinomial(n=np.round(Nd[i]), pvals=prob_by_class[i]) # Remind: this could actually go beyond Nb[i, j] for some (i, j)
            
        # Ds is now Deterministic:
        Ds = np.tile(np.round(Nd), [self.K, 1]).T * population_distr
        
        Pb = np.sum(Y * (Db.T / (np.maximum(1, np.sum(Db, 1))) ) .T, axis=1)
        new_P = self.nu * Pb + (1-self.nu) * Ps
        if np.any(np.isnan(new_P)):
            raise Exception()
        new_M =  M + Db - Ds
        new_R = Ns - np.round(Nd)
        latent = namedtuple("Model",
            "Pb, Ps, Db, Ds, Nb, Ns, Pi, A, AY_unnorm, prob_by_class")\
            (Pb, Ps, Db, Ds, Nb, Ns, Pi, A, AY_unnorm, prob_by_class)
        return (new_P, new_M, new_R, Nd), latent

    def run(self, T, Y=None, initial_M=None, initial_P=None, initial_R=None, seed=123):
        if initial_M is None:
            initial_M = np.tile(self.N * self.Gammak, [self.L, 1])
        if initial_P is None:
            initial_P = np.mean(self.Y @ self.Gammak)
        
        if Y is None:
            Y = self.Y
        
        if Y.shape == (self.K, ):
            Y = np.tile(Y, [T, 1])
        
        assert Y.shape == (T, self.K)
        
        if initial_R is None:
            initial_R = np.zeros(self.L)
        
        P = [initial_P * np.ones(self.L)]
        M = [initial_M]
        R = [initial_R]
        Nd = [np.zeros(self.L)]
        latents = []

        np.random.seed(seed)
        for t in range(1,T):
            (new_P, new_M, new_R, new_Nd), latent = self.run_one_timestep(
                P[-1], M[-1], R[-1], Y=Y[t])

            P.append(new_P)
            M.append(new_M)
            R.append(new_R)
            Nd.append(new_Nd)
            latents.append(latent)

        P = np.array(P)
        M = np.array(M)
        R = np.array(R)
        Nd = np.array(Nd)
        return P, M, R, Nd, latents
        
