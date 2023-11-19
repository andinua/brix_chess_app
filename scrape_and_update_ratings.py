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
from plotly.subplots import make_subplots
import pydeck as pdk
import re

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
    "https://swissonlinetournament.com/Tournament/Details/24783a7442cc4666ad4350fd8e47f617?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/6c5b336f3b244b8992ff76b0dce1cc18?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/c8980bff2afb4b2abf50e9195e44b3bb?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/3bec293b0c41400683f477705644af05?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/77a71054d024477394972ba6758bd6f5?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/c29bda4bbf3e46eb8db7ff8a4d50c89f?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/c297bf58efab488da043663e96b00125?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/ce1c01baac27464ebf16215d637b9c77?allRounds=true"
]
ROUNDS_FILE = 'round_results.json'
RATINGS_FILE = "player_ratings.json"
H2H_FILE = 'head_to_head.json'
RESULTS_FILE = 'player_results.json'
STANDINGS_FILE = 'standings.json'
GEOGRAPHY_FILE = 'geography.csv'

with open('aliases.json', "r") as file:
    ALIASES = json.load(file)

#TODO: unify all player names and results with alises
#TODO: visualize on a map all countries of representation
#TODO: Round all ratings to integer before plotting
#TODO: Need to deal with Bye matches (opponent == "")
#TODO: also extract and store the tournament name and date from "My Tournaments page"
#TODO: add a tooltip or short explanation of how the Bucholz/tiebreaker scores work
#TODO: incorporate registration with drop down menu (player ids/player list)
#TODO: Add Last updated on: 


def get_alias(player_name):
    if player_name in ALIASES:
        return ALIASES[player_name]
    else:
        return player_name


def scrape_results(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    rounds = []

    for table in soup.findAll("table", class_="table pairs-table"):
        round_results = []
        for row in table.findAll("tr", class_="result-row"):
            cells = row.findAll("td")
            white_player = get_alias(cells[1].get_text(strip=True))
            black_player = get_alias(cells[5].get_text(strip=True))
            result = cells[3].get_text(strip=True)

            round_results.append((white_player, black_player, result))

        rounds.append(round_results)

    return rounds


def scrape_standings(url):
    # Modify the URL as needed
    url = url.split("?")[0].replace("Details","Rating")

    # Fetch the page content
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    # Find the table by its class
    table = soup.find('table', class_='table table-striped rating-table')

    # Identify the number of rounds from the table header
    header = table.find('thead').find_all('th')
    num_rounds = len(header) - 5  # Subtract fixed columns: Position, Name, Points, Buc1, BucT

    # Initialize an empty list to store each player's data
    standings = []

    # Iterate over table rows (skip the header row)
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')

        # Extract data from each cell and add it to the dictionary
        player_data = {
            'Position': cols[0].text.strip(),
            'Name': get_alias(cols[1].text.strip()),
            'Points': cols[2].text.strip(),
        }

        # Dynamically add rounds data
        for i in range(num_rounds):
            round_key = f'Round #{i+1}'
            player_data[round_key] = cols[i + 3].text.strip()  # Adjust index to skip fixed columns

        # Add Buc1 and BucT
        player_data['Buc1'] = cols[-2].text.strip()
        player_data['BucT'] = cols[-1].text.strip()

        standings.append(player_data)

    return standings


# Load the standings data
def load_standings():
    try:
        with open(STANDINGS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def display_standings():
    standings_data = load_standings()
    if standings_data:
        # Creating a mapping between user-friendly names and actual keys
        tournament_keys = list(standings_data.keys())
        tournament_name_to_key = {f'Tournament {i+1}': key for i, key in enumerate(tournament_keys)}

        # Displaying user-friendly names in the select box
        selected_tournament_name = st.selectbox("Select a Tournament", list(tournament_name_to_key.keys()))

        # Finding the actual key from the selected name
        selected_tournament_key = tournament_name_to_key[selected_tournament_name]

        # Convert the selected standings to a DataFrame for better visualization
        standings_df = pd.DataFrame(standings_data[selected_tournament_key])
        standings_df.set_index(standings_df.columns[0], inplace=True)
        st.table(standings_df)
    else:
        st.write("No tournament standings data available.")


def aggregate_results(results_data):
    # Aggregating W-D-L results for each player
    aggregated_results = {}
    for player, results in results_data.items():
        total_wins = results["white"]["W"] + results["black"]["W"]
        total_draws = results["white"]["D"] + results["black"]["D"]
        total_losses = results["white"]["L"] + results["black"]["L"]
        aggregated_results[player] = {'Wins': total_wins, 'Draws': total_draws, 'Losses': total_losses}
    return aggregated_results


def display_ranking():
    try:
        with open(RATINGS_FILE, "r") as file:
            ratings_data = json.load(file)

        with open(RESULTS_FILE, "r") as file:
            results_data = json.load(file)

        # Convert to DataFrame and rename columns
        ratings_df = pd.DataFrame.from_dict(ratings_data, orient='index')
        ratings_df.rename(columns={"rating": "Rating", "rd": "Rating Uncertainty (+/-)"}, inplace=True)

        # Convert 'Rating' and 'Rating Uncertainty' to integers
        ratings_df['Rating'] = ratings_df['Rating'].astype(int)
        ratings_df['Rating Uncertainty (+/-)'] = ratings_df['Rating Uncertainty (+/-)'].astype(int)

        # Aggregate results and create a DataFrame
        aggregated_results = aggregate_results(results_data)
        results_df = pd.DataFrame.from_dict(aggregated_results, orient='index')

        # Merge the ratings and results data
        combined_df = ratings_df.merge(results_df, left_index=True, right_index=True)

        # Sort by rating in descending order
        sorted_combined_df = combined_df.sort_values(by='Rating', ascending=False)

        # Reset index to add a numerical index (Rank)
        sorted_combined_df.reset_index(inplace=True)
        sorted_combined_df.index = sorted_combined_df.index + 1
        sorted_combined_df.rename(columns={"index": "Player"}, inplace=True)

        # Use the full screen width for the table
        st.table(sorted_combined_df)

    except (FileNotFoundError, json.JSONDecodeError):
        st.write("No ratings or results data available.")


def check_new_tournament(url):
    # Load existing results
    try:
        with open(ROUNDS_FILE, "r") as f:
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
    

def save_results_to_json(url, results, file):
    # Load existing results
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Use today's date as the tournament key
    key = url.split("/")[-1].split("?")[0]
    if key not in data:
        data[key] = results
        with open(file, "w") as f:
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

    with open(ROUNDS_FILE, "r") as f:
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
    outcome_colors = ['green', 'gray', 'red']  # Colors for each outcome

    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

    for i, color in enumerate(colors):
        total_games = sum([player_results[color][outcome] for outcome in outcomes])
        if total_games == 0:  # Avoid division by zero
            continue

        values = [player_results[color][outcome] for outcome in outcomes]
        percentages = [val / total_games * 100 for val in values]

        # Doughnut chart for each color
        fig.add_trace(go.Pie(
            labels=outcome_labels,
            values=values,
            name=f'{color.capitalize()} pieces',
            hole=0.4,
            marker_colors=outcome_colors,
            hoverinfo='label+percent+value',
            hovertemplate='%{label}: %{value} games (%{percent})<extra></extra>'
        ), 1, i + 1)  # Position the chart in the subplot

    # Update chart layout
    fig.update_layout(
        title_text=f'Game Outcomes for {player_name} by Color',
        annotations=[
            dict(text='White', x=0.20, y=0.5, font_size=20, showarrow=False),
            dict(text='Black', x=0.80, y=0.5, font_size=20, showarrow=False)
        ],
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def clean_country_data(df):
    country_corrections = {
        "Italian": "Italy",
        "UK": "United Kingdom",
        "Czech": "Czech Republic",
        "KAZ": "Kazakhstan",
        "CZE": "Czech Republic",
        "Fr": "France",
        "Brazilian": "Brazil",
        "Slovak": "Slovakia",
        "Ukrainian": "Ukraine",
        "Gondorian": None,  # Assuming 'Gondorian' is a fictional place and should be removed
        "Czech/German": "Czech Republic",  # Assuming preference for Czech Republic
        # Add more corrections as necessary...
    }
    df['country'] = df['country'].replace(country_corrections)
    df = df.dropna()  # Drop rows with NaN values, which may result from non-country entries
    return df


def get_lat_lon(country, cc):
    # Find the row in the dataset where the country matches
    row = cc[cc['name'] == country]
    if not row.empty:
        return row.iloc[0]['latitude'], row.iloc[0]['longitude']
    else:
        return None, None


def load_geo_data():
    df = clean_country_data(pd.read_csv(GEOGRAPHY_FILE, encoding='ISO-8859-1'))
    cc = pd.read_csv('countries.csv', encoding='ISO-8859-1')
    df = df.drop_duplicates(subset=['name', 'country'])
    
    # Apply get_lat_lon function with the country coordinates dataframe
    df['lat'], df['lon'] = zip(*df['country'].apply(lambda x: get_lat_lon(x, cc)))
    
    return df.dropna(subset=['lat', 'lon'])


def map_visualization(df):
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=df['lat'].mean(),
            longitude=df['lon'].mean(),
            zoom=1,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=df,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200000,
            ),
        ],
    ))
    

def format_values(x):
    if np.isnan(x):
        return x
    return int(round(x))


# Extract the numerical part and use it for sorting
def extract_number(s):
    match = re.search(r'\d+$', s)
    return int(match.group()) if match else 0


def plot_player_evolution_streamlit(player_name, df):
    # Visualize medals and best placement
    st.header(f"Medals and best placement for {player_name}:")
    standings_data = load_standings()

    # Count podium finishes
    gold_count = 0
    silver_count = 0
    bronze_count = 0
    best_placement = float('inf')

    for tournament in standings_data.values():
        for player_data in tournament:
            if player_data['Name'] == player_name:
                position = player_data['Position']
                # Updating best_placement logic if necessary
                if '-' in position:
                    pos_value = int(position.split('-')[0])  # Take the lower bound for tied positions
                    best_placement = min(best_placement, pos_value)
                else:
                    best_placement = min(best_placement, int(position))

                if '1-' in position or position == '1':
                    gold_count += 1
                elif '2-' in position or position == '2':
                    silver_count += 1
                elif '3-' in position  or position == '3':
                    bronze_count += 1

    # Display podium finishes or best placement with larger emojis
    if gold_count or silver_count or bronze_count:
        medals_html = f"<span style='font-size: 30px;'>🥇 x {gold_count} 🥈 x {silver_count} 🥉 x {bronze_count}</span>"
        st.markdown(medals_html, unsafe_allow_html=True)
    else:
        if best_placement < float('inf'):
            st.write(f"Best placement: {best_placement}")
        else:
            st.write("No placements available.")

    # Visualize medals and best placement
    st.header(f"Rating evolution in time for {player_name}:")
    all_tournaments = sorted({col.split('-')[0] for col in df.columns}, key=extract_number)

    end_of_tournament_ratings = []
    max_ratings = []
    round_ratings = []
    hover_texts = []
    round_tournaments = []

    last_known_rating = None
    player_data = df.loc[player_name].apply(format_values)

    for tournament in all_tournaments:
        tournament_data = player_data[player_data.index.str.startswith(tournament + '-')]

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
        
        # Sort opponents alphabetically
        sorted_opponents = sorted(opponents)

        # Create a drop-down list of opponents using selectbox, now with sorted opponents
        selected_opponent = st.selectbox("Select an opponent", sorted_opponents)

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
            standings = scrape_standings(url)
            save_results_to_json(url, results, ROUNDS_FILE)
            save_results_to_json(url, standings, STANDINGS_FILE)
            update_ratings_dataframe()
    # plot_player_evolution()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Player Performance", "Tournament Standings", "Global ranking", "Player geography"])

    # Tab for Player Performance
    with tab1:
        st.title('Player Performance')
        df = pd.read_csv('player_ratings.csv', index_col=0)

        # Sort players alphabetically
        sorted_df = df.sort_index()

        # Find the position of 'Andrei Dinu' in the sorted index
        default_index = sorted_df.index.get_loc('Andrei Dinu') if 'Andrei Dinu' in sorted_df.index else 0

        # Creating a select box for players with 'Andrei' as the default selection
        selected_player = st.selectbox('Select a player:', sorted_df.index, index=default_index)

        # Plotting the selected player's data
        plot_player_evolution_streamlit(selected_player, sorted_df)

    # Tab for Tournament Standings
    with tab2:
        st.title('Tournament Standings')
        display_standings()

    # Tab for Global Ranking
    with tab3:
        st.title('Global Ranking')
        display_ranking()

    # Tab for Player geography    
    with tab4:
        st.title('Country Distribution of Players')
        map_visualization(load_geo_data())


if __name__ == '__main__':
    main()