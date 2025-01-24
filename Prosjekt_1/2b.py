import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Define the slot machine payout probabilities and rewards
payouts = {
    "Br/Br/Br": (1 / 64, 20),
    "Bl/Bl/Bl": (1 / 64, 15),
    "L/L/L":    (1 / 64, 5),
    "C/C/C":    (1 / 64, 3),
    "C/C/?":    (3 / 64, 2),
    "C/?/?":    (9 / 64, 1),
}

# Simulate the slot machine
def simulate_slot_machine(initial_coins, simulations):
    results = []

    for _ in range(simulations):
        coins = initial_coins
        plays = 0

        while coins > 0:
            coins -= 1  # Deduct 1 coin per play
            plays += 1

            # Simulate the slot machine
            spin = random.random()
            cumulative_probability = 0

            # We check if we get any of the rewards
            for outcome, (probability, reward) in payouts.items():
                cumulative_probability += probability
                if spin <= cumulative_probability:
                    coins += reward
                    break

        results.append(plays)

    return results

# Parameters
initial_coins = 10
simulations = 10000

# Run the simulation
results = simulate_slot_machine(initial_coins, simulations)

# Calculate mean and median
mean_plays = np.mean(results)
median_plays = np.median(results)

# Print results
print(f"Mean plays until broke: {mean_plays}")
print(f"Median plays until broke: {median_plays}")

# Plot results as a kde
plt.figure(figsize=(10, 6))
kde = gaussian_kde(results)
x_values = np.linspace(min(results), max(results), 500)
y_values = kde(x_values)
plt.plot(x_values, y_values, color="blue", label="Distribution (KDE)", linewidth=2)
plt.axvline(mean_plays, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean_plays:.2f}")
plt.axvline(median_plays, color="green", linestyle="dashed", linewidth=1.5, label=f"Median: {median_plays:.2f}")
plt.title("Distribution of Plays Until Broke (KDE)", fontsize=16)
plt.xlabel("Number of Plays", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
