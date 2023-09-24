"""
Simulate the evolution of players' ratings based on a series of chess matches. Steps:

1.Create a few Player objects representing different players.
2.Simulate some chess matches between these players.
3.Update the ratings based on match outcomes.
4.Plot the evolution of each player's rating over time.
"""

import random
import matplotlib.pyplot as plt
from glicko import Player

def simulate_matches(num_matches=10):
    player1 = Player()
    player2 = Player()

    results = []

    for i in range(num_matches):
        outcome = random.choice([0, 0.5, 1])  # random outcome: 0 (loss), 0.5 (draw), 1 (win)
        results.append(outcome)

    player1_ratings = [player1.rating]  # Store initial rating
    player2_ratings = [player2.rating]  # Store initial rating

    for outcome in results:
        if outcome == 0.5:
            player1_outcome = 0.5
            player2_outcome = 0.5
        elif outcome == 1:
            player1_outcome = 1
            player2_outcome = 0
        else:
            player1_outcome = 0
            player2_outcome = 1

        player1.update_player([player2.rating], [player2.rd], [player1_outcome])
        player2.update_player([player1.rating], [player1.rd], [player2_outcome])

        player1_ratings.append(player1.rating)
        player2_ratings.append(player2.rating)

    return player1_ratings, player2_ratings

def plot_ratings(player1_ratings, player2_ratings):
    # Plotting
    plt.plot(player1_ratings, label="Player 1", marker='o')
    plt.plot(player2_ratings, label="Player 2", marker='o')
    plt.xlabel('Match Number')
    plt.ylabel('Rating')
    plt.title('Rating Evolution over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    player1_ratings, player2_ratings = simulate_matches()
    plot_ratings(player1_ratings, player2_ratings)