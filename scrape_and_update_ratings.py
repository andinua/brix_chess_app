"""
Scrape, calculate, plot
"""

import requests
from bs4 import BeautifulSoup
from glicko import Player
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np
import streamlit as st

# everything should be stored and the loaded for plotting; there should be no updating for tournaments already included in the rating

new_urls = [
    "https://swissonlinetournament.com/Tournament/Details/487026ec898141229c79f424d4a91158?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/871e0b55b93d427a8f024f32a0e0dd3f?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/66c4e5add2e645218fd742c148a12c49?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/1e9c1999c93343909606f587f4aad32a?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/51bedead000741a4b1fdc2139287593b?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/0f2736a25a5f43bd804cdae4d383d95f?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/130cbc69c4fc4eea9747aa65bb33f985?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/9550daf89e30439580f4ea1270c3c20c?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/ea2474d6681f48f288d33163b6c5b9d3?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/24783a7442cc4666ad4350fd8e47f617?allRounds=true"
]
JSON_FILE = "round_results.json"
RATINGS_FILE = "player_ratings.json"

#TODO: On the player page, selecting a 1 v 1 matchup will plot the player evolutions in the same graph, comparison

def scrape_results(url):
    #TODO: also extract and store the tournament name and date
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    rounds = []

    for table in soup.findAll("table", class_="table pairs-table"):
        round_results = []
        for row in table.findAll("tr", class_="result-row"):
            cells = row.findAll("td")
            white_player = cells[1].get_text(strip=True)
            black_player = cells[5].get_text(strip=True)
            result = cells[3].get_text(strip=True)

            round_results.append((white_player, black_player, result))

        rounds.append(round_results)

    return rounds


def check_new_tournament(url):
    # Load existing results
    try:
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Use today's date as the tournament key
    key = url.split("/")[-1].split("?")[0]
    if key not in data:
        return True
    else:
        print("Results for today's tournament already added.")
        return False
    

def save_results_to_json(url, results):
    # Load existing results
    try:
        with open(JSON_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Use today's date as the tournament key
    key = url.split("/")[-1].split("?")[0]
    if key not in data:
        data[key] = results
        with open(JSON_FILE, "w") as f:
            json.dump(data, f)


def load_ratings():
    try:
        with open(RATINGS_FILE, "r") as f:
            ratings_data = json.load(f)
            return {
                k: Player(rating=v["rating"], rd=v["rd"])
                for k, v in ratings_data.items()
            }
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_ratings(players):
    with open(RATINGS_FILE, "w") as f:
        data = {k: {"rating": v.rating, "rd": v.rd} for k, v in players.items()}
        json.dump(data, f)


def update_ratings_dataframe():
    # Check if the CSV file exists and initialize/load the DataFrame accordingly
    if os.path.exists('player_ratings.csv'):
        df = pd.read_csv('player_ratings.csv', index_col=0)
    else:
        df = pd.DataFrame()

    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    players = load_ratings()

    # Assuming the last key in data is the new tournament
    new_tournament_key = list(data.keys())[-1]
    new_tournament = data[new_tournament_key]

    for round_index, round_results in enumerate(new_tournament):
        for white, black, result in round_results:
            if result == "1-0":
                white_result, black_result = 1, 0
            elif result == "0-1":
                white_result, black_result = 0, 1
            else:
                white_result, black_result = 0.5, 0.5
                
            # Ensure players exist in the players dictionary
            # If they don't exist, they'll be initialized with default values.
            if white not in players:
                players[white] = Player()
            if black not in players:
                players[black] = Player()

            players[white].update_player([players[black].rating], [players[black].rd], [white_result])
            players[black].update_player([players[white].rating], [players[white].rd], [black_result])

            # Update dataframe with the new ratings for the new tournament
            col_name_white = f't{len(data)}-r{round_index + 1}'
            df.loc[white, col_name_white] = players[white].rating
            df.loc[black, col_name_white] = players[black].rating

    save_ratings(players)
    df.to_csv('player_ratings.csv')  # Save the updated dataframe to the same CSV file


def plot_player_evolution():
    df = pd.read_csv('player_ratings.csv', index_col=0)
    
    # Identify all available tournaments from the dataframe
    all_tournaments = sorted({col.split('-')[0] for col in df.columns})

    for player_name in df.index:
        plt.figure(figsize=(10, 6))
        
        # Initialize lists for end of tournament and max ratings
        end_of_tournament_ratings = []
        max_ratings = []
        last_known_rating = None  # This will keep track of the last known rating

        # Extract data for the player
        player_data = df.loc[player_name]

        for tournament in all_tournaments:
            # Filtering data for the specific tournament
            tournament_data = player_data.filter(like=tournament)
            
            # Check if player has data for the tournament
            if not tournament_data.isna().all():
                tournament_ticks = [tournament] * len(tournament_data.dropna())
                if tournament != 't1':
                    plt.scatter(tournament_ticks, tournament_data.dropna().values, color='red', s=15)  # Plot round ratings
                else:
                    # Don't plot round rating and max rating for first tournament, high volatility increases y range
                    plt.scatter([tournament], [last_known_rating], color='red', s=0)  # Leave empty space
                
                last_known_rating = tournament_data.dropna().values[-1]
                end_of_tournament_ratings.append(last_known_rating)
                max_ratings.append(tournament_data.dropna().values.max() if tournament != 't1' else np.nan)
            else:
                # Player did not participate in this tournament
                end_of_tournament_ratings.append(last_known_rating)  # Use the last known rating if available
                max_ratings.append(np.nan)
                plt.scatter([tournament], [last_known_rating], color='red', s=0)  # Leave empty space

        # Plot the end of tournament ratings and max ratings
        plt.plot(all_tournaments, end_of_tournament_ratings, label='End of Tournament', marker='o', linestyle='-')
        plt.scatter(all_tournaments, max_ratings, color='green', s=50, marker='*', label='Max Rating in Tournament')

        # Settings for the x-axis
        plt.xticks(all_tournaments, [f'Tournament {i}' for i in range(1, len(all_tournaments)+1)], rotation=45)

        # Title, labels, and other configurations
        plt.xlabel('Tournaments')
        plt.ylabel('Rating')
        plt.title(f'Rating Evolution for {player_name}')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()


def plot_player_evolution_streamlit(player_name, df):
    all_tournaments = sorted({col.split('-')[0] for col in df.columns})

    plt.figure(figsize=(10, 6))

    end_of_tournament_ratings = []
    max_ratings = []
    last_known_rating = None

    player_data = df.loc[player_name]

    for tournament in all_tournaments:
        tournament_data = player_data.filter(like=tournament)

        if not tournament_data.isna().all():
            tournament_ticks = [tournament] * len(tournament_data.dropna())
            if tournament != 't1':
                plt.scatter(tournament_ticks, tournament_data.dropna().values, color='red', s=15)
            else:
                plt.scatter([tournament], [last_known_rating], color='red', s=0)

            last_known_rating = tournament_data.dropna().values[-1]
            end_of_tournament_ratings.append(last_known_rating)
            max_ratings.append(tournament_data.dropna().values.max() if tournament != 't1' else np.nan)
        else:
            end_of_tournament_ratings.append(last_known_rating)
            max_ratings.append(np.nan)
            plt.scatter([tournament], [last_known_rating], color='red', s=0)

    plt.plot(all_tournaments, end_of_tournament_ratings, label='End of Tournament', marker='o', linestyle='-')
    plt.scatter(all_tournaments, max_ratings, color='green', s=50, marker='*', label='Max Rating in Tournament')

    plt.xticks(all_tournaments, [f'Tournament {i}' for i in range(1, len(all_tournaments)+1)], rotation=45)
    plt.xlabel('Tournaments')
    plt.ylabel('Rating')
    plt.title(f'Rating Evolution for {player_name}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    st.pyplot(plt)


def main():
    for url in new_urls:
        if check_new_tournament(url):
            results = scrape_results(url)
            save_results_to_json(url, results)
            update_ratings_dataframe()
    # plot_player_evolution()

    st.title('Player Rating Evolution')
    df = pd.read_csv('player_ratings.csv', index_col=0)

    # Creating a select box for players
    selected_player = st.selectbox('Select a player:', df.index)

    # Plotting the selected player's data
    plot_player_evolution_streamlit(selected_player, df)


if __name__ == '__main__':
    main()