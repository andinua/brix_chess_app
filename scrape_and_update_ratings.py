"""
Scrape, calculate, plot
"""

import requests
from bs4 import BeautifulSoup
from glicko import Player
import matplotlib.pyplot as plt
import json

# TODO: Currently ratings are stored and then recalculated for each tournament, 
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
]
JSON_FILE = "round_results.json"
RATINGS_FILE = "player_ratings.json"


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


def plot_player_evolution():
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    players = load_ratings()
    players_ratings = {}

    tournament_lens = [len(tournament) for tournament_key, tournament in data.items()]

    player_tournament_mapping = {}  # This dictionary maps a player to the tournaments they participated in

    for tournament_index, (tournament_key, tournament) in enumerate(data.items()):
        for round_results in tournament:
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

                if white not in players_ratings:
                    players_ratings[white] = [players[white].rating]
                else:
                    players_ratings[white].append(players[white].rating)

                if black not in players_ratings:
                    players_ratings[black] = [players[black].rating]
                else:
                    players_ratings[black].append(players[black].rating)

                # Map players to tournaments they participated in
                if white not in player_tournament_mapping:
                    player_tournament_mapping[white] = []
                if black not in player_tournament_mapping:
                    player_tournament_mapping[black] = []

                if tournament_index not in player_tournament_mapping[white]:
                    player_tournament_mapping[white].append(tournament_index)
                if tournament_index not in player_tournament_mapping[black]:
                    player_tournament_mapping[black].append(tournament_index)

    save_ratings(players)

    for player_name, player_tournaments in player_tournament_mapping.items():
        plt.figure(figsize=(10, 6))
        end_of_tournament_ratings = []

        # Plot the rating evolution across tournaments with a line plot
        for t_index in player_tournaments:
            t_length = tournament_lens[t_index]
            start_index = sum(tournament_lens[i] for i in player_tournaments if i < t_index)
            end_index = start_index + t_length

            # Ensure ratings exist for the given range
            if start_index < len(players_ratings[player_name]) and end_index <= len(players_ratings[player_name]):
                # Plot round ratings for the current tournament
                round_ticks = [t_index] * t_length
                plt.scatter(round_ticks, players_ratings[player_name][start_index:end_index], color='red', s=15)

                # Update end of tournament ratings
                if end_index <= len(players_ratings[player_name]):
                    end_of_tournament_ratings.append(players_ratings[player_name][end_index - 1])
                else:
                    end_of_tournament_ratings.append(players_ratings[player_name][-1])

        plt.plot(player_tournaments, end_of_tournament_ratings, label='End of Tournament', marker='o', linestyle='-')
        
        # Highlight the maximum rating for each tournament with a green dot
        max_ratings = []
        for t_index in player_tournaments:
            t_length = tournament_lens[t_index]
            start_index = sum(tournament_lens[i] for i in player_tournaments if i < t_index)
            end_index = start_index + t_length

            if start_index < len(players_ratings[player_name]) and end_index <= len(players_ratings[player_name]):
                max_ratings.append(max(players_ratings[player_name][start_index:end_index]))
            else:
                max_ratings.append(players_ratings[player_name][-1])

        plt.scatter(player_tournaments, max_ratings, color='green', s=50, marker='*', label='Max Rating in Tournament')

        plt.xlabel('Tournaments')
        plt.ylabel('Rating')
        plt.title(f'Rating Evolution for {player_name}')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.xticks(player_tournaments, [f'Tournament {i+1}' for i in player_tournaments])
        plt.show()

if __name__ == "__main__":
    for url in new_urls:
        if check_new_tournament(url):
            results = scrape_results(url)
            save_results_to_json(url, results)
    plot_player_evolution()