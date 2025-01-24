import random

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

# Task 3.1
# We want to simulate the chance of 10 people having the same birthday
print("The chance of 10 people sharing the same birthday is", birthday_simulation(10)) 