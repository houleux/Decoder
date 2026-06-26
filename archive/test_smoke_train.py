import numpy as np
import scipy.sparse as sp
import pandas as pd
from RL.dyna_reldec import LDPCEnvironment, ReldecAgent, train_reldec

def main():
    print("Creating dummy sparse matrix...")
    m, n = 48, 96
    h_csr = sp.random(m, n, density=0.1, format='csr', data_rvs=np.ones, dtype=np.uint8)
    
    print("Initializing LDPCEnvironment...")
    env = LDPCEnvironment(h_csr.toarray(), z=1)
    
    print("Initializing ReldecAgent...")
    m, n = h_csr.shape
    agent = ReldecAgent(z=1, num_cns=m, epsilon=0.1, alpha=0.1, gamma=0.99)
    
    print("Starting smoke training run for 1 episode...")
    # This will call env.reset(), env.step(), etc.
    train_reldec(env, agent, num_episodes=1, l_max=10)
    print("Training complete!")

if __name__ == "__main__":
    main()
