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
import plotly.graph_objects as go

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
RESULTS_FILE = 'player_results.json'

#TODO: Order players alphabetically
#TODO: scrape standings and visualize, for each player - g x gold medals, s x silver, b x bronze or best ranking if never on podium
#TODO: visualize on a map all countries of representation
#TODO: Round all ratings to integer before plotting
#TODO: Need to deal with Bye matches (opponent == "")

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


def load_results_by_color():
    try:
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_results_by_color(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)


def save_ratings(players):
    with open(RATINGS_FILE, "w") as f:
        data = {k: {"rating": v.rating, "rd": v.rd} for k, v in players.items()}
        json.dump(data, f)

def load_h2h():
    try:
        with open(H2H_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_h2h(h2h):
    with open(H2H_FILE, "w") as f:
        json.dump(h2h, f)


def update_h2h(h2h, white, black, result):
    def ensure_player_opponent(player, opponent):
        if player not in h2h:
            h2h[player] = {}
        if opponent not in h2h[player]:
            h2h[player][opponent] = {'win': 0, 'loss': 0, 'draw': 0}

    ensure_player_opponent(white, black)
    ensure_player_opponent(black, white)

    if result == "1-0":
        h2h[white][black]['win'] += 1
        h2h[black][white]['loss'] += 1
    elif result == "0-1":
        h2h[white][black]['loss'] += 1
        h2h[black][white]['win'] += 1
    else:
        h2h[white][black]['draw'] += 1
        h2h[black][white]['draw'] += 1


def update_ratings_dataframe():
    # Check if the CSV file exists and initialize/load the DataFrame accordingly
    if os.path.exists('player_ratings.csv'):
        df = pd.read_csv('player_ratings.csv', index_col=0)
    else:
        df = pd.DataFrame()

    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    results_by_color = load_results_by_color()  # Load results from the JSON file

    players = load_ratings()
    h2h = load_h2h()  # Load H2H data at the start

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
            update_h2h(h2h, white, black, result)

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

            # Update the results_by_color for each player based on their color and game outcome
            if white not in results_by_color:
                results_by_color[white] = {"white": {"W": 0, "D": 0, "L": 0}, "black": {"W": 0, "D": 0, "L": 0}}
            if black not in results_by_color:
                results_by_color[black] = {"white": {"W": 0, "D": 0, "L": 0}, "black": {"W": 0, "D": 0, "L": 0}}
            
            if result == "1-0":
                results_by_color[white]["white"]["W"] += 1
                results_by_color[black]["black"]["L"] += 1
            elif result == "0-1":
                results_by_color[white]["white"]["L"] += 1
                results_by_color[black]["black"]["W"] += 1
            else:
                results_by_color[white]["white"]["D"] += 1
                results_by_color[black]["black"]["D"] += 1

    save_h2h(h2h)
    save_ratings(players)
    save_results_by_color(results_by_color)
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


def plot_player_results_by_color(player_name):
    # Load player results
    with open(RESULTS_FILE, "r") as f:
        results_data = json.load(f)

    player_results = results_data.get(player_name, None)
    if not player_results:
        st.write(f"No results found for {player_name}.")
        return

    colors = ['white', 'black']
    outcomes = ['W', 'D', 'L']
    outcome_labels = ['Win', 'Draw', 'Loss']
    outcome_colors = ['green', 'gray', 'red']  # Adding colors for each outcome

    bar_width = 0.2
    bar_positions = np.arange(len(colors))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, outcome in enumerate(outcomes):
        results = [player_results[color][outcome] for color in colors]
        ax.bar(bar_positions + i * bar_width, results, width=bar_width, label=outcome_labels[i], color=outcome_colors[i])

    ax.set_xlabel('Color')
    ax.set_ylabel('Number of Games')
    ax.set_title(f'Game Outcomes for {player_name} by Color')
    ax.set_xticks(bar_positions + bar_width)
    ax.set_xticklabels(colors)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

def format_values(x):
    if np.isnan(x):
        return x
    return int(round(x))

def plot_player_evolution_streamlit(player_name, df):
    all_tournaments = sorted({col.split('-')[0] for col in df.columns})

    end_of_tournament_ratings = []
    max_ratings = []
    round_ratings = []
    hover_texts = []
    round_tournaments = []

    last_known_rating = None
    player_data = df.loc[player_name].apply(format_values)

    for tournament in all_tournaments:
        tournament_data = player_data.filter(like=tournament)

        if not tournament_data.isna().all():
            if tournament != 't1':
                round_tournament_ticks = [tournament] * len(tournament_data.dropna())
                round_ratings.extend(tournament_data.dropna().values)
                hover_texts.extend([f"Round {idx + 1} Rating: {rating}" for idx, rating in enumerate(tournament_data.dropna().values)])
                round_tournaments.extend(round_tournament_ticks)
            else:
                # If it's 't1', we consider the last known rating which is initialized to None
                round_ratings.append(None)
                hover_texts.append(None)
                round_tournaments.append(tournament)

            last_known_rating = tournament_data.dropna().values[-1]
            end_of_tournament_ratings.append(last_known_rating)
            max_ratings.append(tournament_data.dropna().values.max() if tournament != 't1' else np.nan)
        else:
            end_of_tournament_ratings.append(last_known_rating)
            max_ratings.append(np.nan)

    fig = go.Figure()

    # Plotting end of tournament ratings
    fig.add_trace(go.Scatter(
        x=all_tournaments,
        y=end_of_tournament_ratings,
        mode='lines+markers',
        name='End of Tournament',
        marker=dict(symbol='square', size=15),
        hovertemplate='Tournament: %{x}<br>End Rating: %{y}'
    ))

    # Plotting round ratings, with handling of 't1' and non-available data
    fig.add_trace(go.Scatter(
        x=round_tournaments,
        y=round_ratings,
        mode='markers',
        name='Round Rating',
        marker=dict(color='red', size=5),
        hoverinfo='text',
        hovertext=hover_texts
    ))

    # Plotting max ratings
    fig.add_trace(go.Scatter(
        x=all_tournaments,
        y=max_ratings,
        mode='markers',
        name='Max Rating in Tournament',
        marker=dict(color='green', symbol='star', size=10),
        hovertemplate='Tournament: %{x}<br>Max Rating: %{y}'
    ))

    # Filtering out the 'None' entries for 't1' tournament
    fig.for_each_trace(lambda trace: trace.update(marker=dict(size=[0 if y is None else (trace.marker.size if hasattr(trace.marker, 'size') and trace.marker.size is not None else 5) for y in trace.y])))

    fig.update_layout(
        title=f'Rating Evolution for {player_name}',
        xaxis_title='Tournaments',
        yaxis_title='Rating',
        xaxis_tickvals=all_tournaments,
        xaxis_ticktext=[f'Tournament {i}' for i in range(1, len(all_tournaments) + 1)]
    )

    st.plotly_chart(fig)

    # Plot player results by color
    st.header(f"Game Outcomes by Color for {player_name}:")
    plot_player_results_by_color(player_name)

    # Load H2H data
    with open(H2H_FILE, "r") as f:
        h2h = json.load(f)

    if player_name in h2h:
        st.header(f"Head to Head Records for {player_name}:")
        
        # Get all opponents for the player from the H2H data
        opponents = list(h2h[player_name].keys())
        
        # Create a drop-down list of opponents using selectbox
        selected_opponent = st.selectbox("Select an opponent", opponents)

        # Display details only if an opponent is selected from the drop-down list
        if selected_opponent:
            if st.button(f"View Rating Evolution with {selected_opponent}"):
                plot_two_player_evolution(player_name, selected_opponent, df)
            
            # Card-like visuals for the selected H2H record
            record = h2h[player_name][selected_opponent]
            col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 2])
            with col1:
                st.write(" ")
            with col2:
                st.subheader(selected_opponent)
            # Determine the maximum value and apply background color accordingly
            max_val = max(record['win'], record['draw'], record['loss'])
            with col3:
                if record['win'] == max_val:
                    st.markdown(f"<div style='background-color: green; padding: 10px; border-radius: 5px;'>Wins: {record['win']}</div>", unsafe_allow_html=True)
                else:
                    st.write(f"Wins: {record['win']}")
            with col4:
                if record['loss'] == max_val:
                    st.markdown(f"<div style='background-color: red; padding: 10px; border-radius: 5px;'>Losses: {record['loss']}</div>", unsafe_allow_html=True)
                else:
                    st.write(f"Losses: {record['loss']}")
            with col5:
                if record['draw'] == max_val:
                    st.markdown(f"<div style='background-color: gray; padding: 10px; border-radius: 5px;'>Draws: {record['draw']}</div>", unsafe_allow_html=True)
                else:
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