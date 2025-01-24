import random
import matplotlib.pyplot as plt

# Function to simulate probability of at least two people sharing a birthday
def birthday_simulation(N, runs=10000):
    successes = 0

    for _ in range(runs):
        # Generate random birthdays for N people
        birthdays = [random.randint(1, 365) for _ in range(N)]
        # Check for duplicates
        if len(birthdays) != len(set(birthdays)):
            successes += 1

    # Probability of at least one duplicate
    return successes / runs

# Used only for task b
def probabilities_in_range(start=10, end=50, target_probability=0.5, trials=10000):
    probabilities = []
    for N in range(start, end + 1):
        prob = birthday_simulation(N, trials)
        probabilities.append((N, prob))

    # Find proportion where probability >= target_probability
    count_above_threshold = sum(1 for _, prob in probabilities if prob >= target_probability)
    proportion = count_above_threshold / (end - start + 1)

    return probabilities, proportion

# Task 3.1
# We want to find the chance that N people contain at least two that share a birthday
print(f"The chance of 25 people contain atleast two people sharing a birthday is", birthday_simulation(10))

# Task (b): Compute probabilities and proportion for N in [10, 50]
results, proportion = probabilities_in_range()

# Print the results
print(f"Proportion of N in [10, 50] with probability >= 50%: {proportion:.4f}")

# Plot the probabilities
Ns, probs = zip(*results)
plt.figure(figsize=(10, 6))
plt.plot(Ns, probs, marker="o", color="blue", label="Probability")
plt.axhline(y=0.5, color="red", linestyle="--", label="50% Threshold")
plt.title("Probability of At Least Two People Sharing a Birthday", fontsize=16)
plt.xlabel("Number of People (N)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.xticks(range(10, 51, 5))
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()
