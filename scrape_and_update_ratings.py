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
H2H_FILE = 'head_to_head.json'

#TODO: Order players alphabetically

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


def update_h2h(white, black, result):
    # Load or initialize the H2H data
    if os.path.exists(H2H_FILE):
        with open(H2H_FILE, "r") as f:
            h2h = json.load(f)
    else:
        h2h = {}

    # Helper function to ensure a player and opponent exists in the H2H data
    def ensure_player_opponent(player, opponent):
        if player not in h2h:
            h2h[player] = {}
        if opponent not in h2h[player]:
            h2h[player][opponent] = {'win': 0, 'loss': 0, 'draw': 0}

    ensure_player_opponent(white, black)
    ensure_player_opponent(black, white)

    # Update the H2H record based on the result
    if result == "1-0":
        h2h[white][black]['win'] += 1
        h2h[black][white]['loss'] += 1
    elif result == "0-1":
        h2h[white][black]['loss'] += 1
        h2h[black][white]['win'] += 1
    else:
        h2h[white][black]['draw'] += 1
        h2h[black][white]['draw'] += 1

    # Save the updated H2H data
    with open(H2H_FILE, "w") as f:
        json.dump(h2h, f)


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
                
            # Update H2H record after processing the game result
            update_h2h(white, black, result)

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


def plot_two_player_evolution(player1, player2, df):
    """Plots the rating evolution for two players."""
    all_tournaments = sorted({col.split('-')[0] for col in df.columns})

    plt.figure(figsize=(10, 6))
    for player_name, color in [(player1, 'blue'), (player2, 'orange')]:
        end_of_tournament_ratings = []
        player_data = df.loc[player_name]
        last_known_rating = None

        for tournament in all_tournaments:
            tournament_data = player_data.filter(like=tournament)
            if not tournament_data.isna().all():
                last_known_rating = tournament_data.dropna().values[-1]
                end_of_tournament_ratings.append(last_known_rating)
            else:
                end_of_tournament_ratings.append(last_known_rating)

        plt.plot(all_tournaments, end_of_tournament_ratings, label=player_name, color=color)

    plt.xticks(all_tournaments, [f'Tournament {i}' for i in range(1, len(all_tournaments)+1)], rotation=45)
    plt.xlabel('Tournaments')
    plt.ylabel('Rating')
    plt.title(f'Rating Evolution for {player1} and {player2}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(plt)


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
    #TODO: add rating after round to legend for red dots, but only once
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    st.pyplot(plt)

    # Load H2H data
    with open(H2H_FILE, "r") as f:
        h2h = json.load(f)

    if player_name in h2h:
        st.header(f"Head to Head Records for {player_name}:")

        for opponent, record in h2h[player_name].items():
            if st.button(f"View Rating Evolution with {opponent}"):
                plot_two_player_evolution(player_name, opponent, df)
            else:
                # Card-like visuals for each H2H record
                col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 2])
                with col1:
                    st.write(" ")
                with col2:
                    st.subheader(opponent)
                with col3:
                    st.write(f"Wins: {record['win']}")
                with col4:
                    st.write(f"Losses: {record['loss']}")
                with col5:
                    st.write(f"Draws: {record['draw']}")
                st.markdown("---")
    else:
        st.write(f"No Head to Head Records found for {player_name}.")


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