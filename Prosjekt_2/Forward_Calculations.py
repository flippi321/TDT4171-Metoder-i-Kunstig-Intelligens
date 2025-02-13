import numpy as np

# Transition probabilities
P_true_given_true = 0.7
P_false_given_true = 0.3
P_true_given_false = 0.3
P_false_given_false = 0.7

# Observation probabilities
P_obs_true_given_true = 0.9
P_obs_false_given_true = 0.1
P_obs_true_given_false = 0.2
P_obs_false_given_false = 0.8

# Initial state probabilities
P_R1_true = 0.5
P_R1_false = 0.5

# Observations (True = 1, False = 0)
observations = [1, 1, 0, 1, 1]

# Forward probabilities initialization
f_true = P_R1_true * (P_obs_true_given_true if observations[0] else P_obs_false_given_true)
f_false = P_R1_false * (P_obs_true_given_false if observations[0] else P_obs_false_given_false)

# Normalize
alpha = f_true + f_false
f_true /= alpha
f_false /= alpha

print(f"Day 1: P(R=True) = {f_true:.4f}, P(R=False) = {f_false:.4f}")

# Forward algorithm for days 2 to 5
for t in range(1, 5):
    obs = observations[t]
    f_t_true = (f_true * P_true_given_true + f_false * P_true_given_false) * (P_obs_true_given_true if obs else P_obs_false_given_true)
    f_t_false = (f_true * P_false_given_true + f_false * P_false_given_false) * (P_obs_true_given_false if obs else P_obs_false_given_false)
    
    # Normalize
    alpha = f_t_true + f_t_false
    f_true = f_t_true / alpha
    f_false = f_t_false / alpha
    
    print(f"Day {t+1}: P(R=True) = {f_true:.4f}, P(R=False) = {f_false:.4f}")
