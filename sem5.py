''' Тут будет комбинация нормального и равномерного распределений '''


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


a = 0
b = 1
rtol = 1e-3

def datageneration():
    tau = 0.5
    mu = 0.5
    sigma = 0.2
    n = 10000
    
    x_N = stats.norm.rvs(loc=mu, scale=sigma, size=int(tau*n))
    x_U = stats.uniform.rvs(loc=a, scale=b-a, size=int((1-tau)*n))
    x = np.concatenate((x_N, x_U))
    return x

    
def p_ij(x, tau, mu, sigma):
    p_iN = stats.norm.pdf(x, loc=mu, scale=sigma) #probabilitu of normal distribution
    p_iU = stats.uniform.pdf(x, loc=a, scale=b-a) #1 / (b-a) not correct for any area, turns to 0 somewhere
    p = tau*p_iN + (1 - tau)*p_iU
    return p, p_iN, p_iU
    
def Tij(x, tau, mu, sigma):
    p, p_iN, p_iU = p_ij(x, tau, mu, sigma)
    T_iN = tau * p_iN / p
    T_iU = (1 - tau) * p_iU / p
    return T_iN, T_iU # np.vstack((T_iN, T_iU)) #склеивае 2 массива


def update_theta(x, *old):
    '''old = (tau, mu, sigma), iterate theta'''
    t_n, t_u = Tij(x, *old)
    tau = np.sum(t_n) / x.size
    mu = np.sum(t_n*x) / np.sum(t_n)
    sigma = np.sqrt(np.sum(t_n * (x - mu)**2) / np.sum(t_n)) #we calculated theme at class
    return tau, mu, sigma

def em(x, tau, mu, sigma, eps=1e-3):
    '''tau, mu, sigma - initial estimations, wait for convergens of three variables'''
    new = (tau, mu, sigma)
    while True:
        old = new
        new = update_theta(x, *old)
        #if np.max(np.abs((new - old) / new)) < rtol:
        if np.allclose(new, old, rtol=rtol, atol=0):
            break

    return new

def main():
    x = datageneration()
    plt.hist(x, bins=100)
    tau, mu, sigma = 0.8, 0.1, 0.1
    #x_ = np.linspace(0, 1, 101)
    #for k in range(6):
    #    plt.plot(x_, 100*p_ij(x_, tau, mu, sigma)[0], label=str(k))
    #    tau, mu, sigma = update_theta(x, tau, mu, sigma)
    #plt.legend()
    tau, mu, sigma = em(x, tau, mu, sigma)
    print(tau, mu, sigma)
    
if __name__ == '__main__':
    main()