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

# Find the smallest N where the probability is a given probability
def find_smallest_N(target_probability=0.5, runs=10000):
    N = 1
    max_runs = 10000

    # We just use N as a shortcut to count runs too
    while N <= max_runs:
        prob = birthday_simulation(N, runs)
        if prob >= target_probability:
            return N, prob
        N += 1

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
# We want to find the smallest number of people, N, where probability of two sharing a birthday is >= 50%
smallest_N, probability_at_smallest_N = find_smallest_N()
print(f"The smallest N where the probability of at least two people sharing a birthday is at least 50% is: {smallest_N} (Probability: {probability_at_smallest_N:.4f})")

# Task (b): Compute probabilities and proportion for N in [10, 50]
results, proportion = probabilities_in_range()

# Print the results
for N, prob in results:
    print(f"N = {N}, Probability = {prob:.4f}")
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
