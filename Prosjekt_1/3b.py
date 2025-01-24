import random
import numpy as np
import matplotlib.pyplot as plt

# Function to simulate the group size required to cover all 365 days by people's birthdays
def simulate_group_size(trials=10000):
    group_sizes = []

    for _ in range(trials):
        days_covered = set()
        group_size = 0

        while len(days_covered) < 365:
            # Add a person with a random birthday
            birthday = random.randint(1, 365)
            days_covered.add(birthday)
            group_size += 1

        group_sizes.append(group_size)

    # Return the group sizes for further analysis
    return group_sizes

# Run the simulation
trials = 10000
group_sizes = simulate_group_size(trials)

# Calculate mean and median
group_size_mean = np.mean(group_sizes)
group_size_median = np.median(group_sizes)

# Print results
print(f"Expected (mean) group size to cover all 365 days: {group_size_mean:.2f}")
print(f"Median group size to cover all 365 days: {group_size_median:.2f}")

# Plotting the distribution of group sizes
plt.figure(figsize=(10, 6))
plt.hist(group_sizes, bins=30, color="skyblue", edgecolor="black", alpha=0.7, density=True)

# Add mean and median lines to the plot
plt.axvline(group_size_mean, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {group_size_mean:.2f}")
plt.axvline(group_size_median, color="green", linestyle="dashed", linewidth=1.5, label=f"Median: {group_size_median:.2f}")

# Customize plot
plt.title("Distribution of Group Sizes to Cover All 365 Days", fontsize=16)
plt.xlabel("Group Size", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Show the plot
plt.show()
