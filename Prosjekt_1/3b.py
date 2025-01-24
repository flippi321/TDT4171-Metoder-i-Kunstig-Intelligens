import random
import numpy as np

# Function to simulate the group size required to cover all 365 days by peoples birthdays
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

    # Return the average group size
    return np.mean(group_sizes)

# Run the simulation
trials = 10000
group_size_mean = simulate_group_size(trials)
print(f"Expected group size to cover all 365 days: {group_size_mean:.2f}")
