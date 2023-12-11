"""
Data Processing: Scrape, calculate, and update data.
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import pandas as pd
from glicko import Player

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
    "https://swissonlinetournament.com/Tournament/Details/ce1c01baac27464ebf16215d637b9c77?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/5aecf13934b6476b8cdfa673a2d737d4?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/a36bea22ea7046d5b6152af88270bf2f?allRounds=true",
    "https://swissonlinetournament.com/Tournament/Details/6db995319eb84dfe94f9f7c9dff70a2f?allRounds=true"
]

ROUNDS_FILE = 'round_results.json'
RATINGS_FILE = "player_ratings.json"
H2H_FILE = 'head_to_head.json'
RESULTS_FILE = 'player_results.json'
STANDINGS_FILE = 'standings.json'

with open('aliases.json', "r", encoding='utf-8') as file:
    ALIASES = json.load(file)

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


def check_new_tournament(url):
    # Load existing results
    try:
        with open(ROUNDS_FILE, "r", encoding='utf-8') as f:
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
        with open(file, "r", encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Use today's date as the tournament key
    key = url.split("/")[-1].split("?")[0]
    if key not in data:
        data[key] = results
        with open(file, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)


def load_ratings():
    try:
        with open(RATINGS_FILE, "r", encoding='utf-8') as f:
            ratings_data = json.load(f)
            return {
                k: Player(rating=v["rating"], rd=v["rd"])
                for k, v in ratings_data.items()
            }
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_results_by_color():
    try:
        with open(RESULTS_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_results_by_color(results):
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def save_ratings(players):
    with open(RATINGS_FILE, "w", encoding='utf-8') as f:
        data = {k: {"rating": v.rating, "rd": v.rd} for k, v in players.items()}
        json.dump(data, f, ensure_ascii=False)

def load_h2h():
    try:
        with open(H2H_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_h2h(h2h):
    with open(H2H_FILE, "w", encoding='utf-8') as f:
        json.dump(h2h, f, ensure_ascii=False)


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

    with open(ROUNDS_FILE, "r", encoding='utf-8') as f:
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


def update_data():
    for idx, url in enumerate(new_urls):
        print(f'Processing tournament {idx + 1} out of {len(new_urls)}')
        if check_new_tournament(url):
            results = scrape_results(url)
            standings = scrape_standings(url)
            save_results_to_json(url, results, ROUNDS_FILE)
            save_results_to_json(url, standings, STANDINGS_FILE)
            update_ratings_dataframe()


if __name__ == "__main__":
    update_data()
