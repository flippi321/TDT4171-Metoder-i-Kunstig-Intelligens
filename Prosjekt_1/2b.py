import random
import numpy as np
import matplotlib.pyplot as plt

# Define the slot machine payout probabilities and rewards
payouts = {
    "BAR/BAR/BAR": (1 / 64, 20),
    "BELL/BELL/BELL": (1 / 64, 15),
    "LEMON/LEMON/LEMON": (1 / 64, 5),
    "CHERRY/CHERRY/CHERRY": (1 / 64, 3),
    "CHERRY/CHERRY/?": (3 / 64, 2),
    "CHERRY/?/?": (9 / 64, 1),
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

# Plotting the results
plt.figure(figsize=(10, 6))
plt.hist(results, bins=30, color="skyblue", edgecolor="black", alpha=0.7, density=True)
plt.axvline(mean_plays, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {mean_plays:.2f}")
plt.axvline(median_plays, color="green", linestyle="dashed", linewidth=1.5, label=f"Median: {median_plays:.2f}")
plt.title("Distribution of Plays Until Broke", fontsize=16)
plt.xlabel("Number of Plays", fontsize=14)
plt.ylabel("Probability Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
