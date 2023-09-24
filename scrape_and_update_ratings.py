"""
Scrape, calculate, plot
"""

import requests
from bs4 import BeautifulSoup
from glicko import Player
import matplotlib.pyplot as plt
import mplfinance.original_flavor as mpf
import matplotlib.dates as mdates
import json

URL = "https://swissonlinetournament.com/Tournament/Details/ea2474d6681f48f288d33163b6c5b9d3?allRounds=true#"
JSON_FILE = 'round_results.json'


def scrape_results(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    
    rounds = []
    
    for table in soup.findAll('table', class_='table pairs-table'):
        round_results = []
        for row in table.findAll('tr', class_='result-row'):
            cells = row.findAll('td')
            white_player = cells[1].get_text(strip=True)
            black_player = cells[5].get_text(strip=True)
            result = cells[3].get_text(strip=True)
            
            round_results.append((white_player, black_player, result))
        
        rounds.append(round_results)
        
    return rounds

def save_results_to_json(results):
    # Load existing results
    try:
        with open(JSON_FILE, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(results)
    
    with open(JSON_FILE, 'w') as f:
        json.dump(data, f)

def plot_player_evolution():
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    # Assume each player starts with a rating of 1500
    players = {}

    for tournament in data:
        for round_results in tournament:
            for white, black, result in round_results:
                if white not in players:
                    players[white] = {'player_obj': Player(), 'ratings': []}
                if black not in players:
                    players[black] = {'player_obj': Player(), 'ratings': []}

                if result == "1-0":
                    white_result, black_result = 1, 0
                elif result == "0-1":
                    white_result, black_result = 0, 1
                else:
                    white_result, black_result = 0.5, 0.5

                players[white]['player_obj'].update_player([players[black]['player_obj'].rating], [players[black]['player_obj'].rd], [white_result])
                players[white]['ratings'].append(players[white]['player_obj'].getRating())
                
                players[black]['player_obj'].update_player([players[white]['player_obj'].rating], [players[white]['player_obj'].rd], [black_result])
                players[black]['ratings'].append(players[black]['player_obj'].getRating())

    tournament_ticks = list(range(1, len(data) + 1))
    for player_name, player_data in players.items():
        plt.figure(figsize=(10,6))
        
        # Extract ratings after each tournament (last rating in each tournament)
        end_of_tournament_ratings = [player_data['ratings'][i * len(tournament) - 1] for i in tournament_ticks]
        
        # Plot the rating evolution across tournaments with a line plot
        plt.plot(tournament_ticks, end_of_tournament_ratings, label='End of Tournament', marker='o', linestyle='-')
        
        # Overlay with a dot plot showing the rating after each round of each tournament
        for t_num, tick in enumerate(tournament_ticks):
            start_index = t_num * len(tournament)
            end_index = (t_num + 1) * len(tournament)
            plt.scatter([tick] * len(tournament), player_data['ratings'][start_index:end_index], color='red', s=15)

        plt.xlabel('Tournaments')
        plt.ylabel('Rating')
        plt.title(f'Rating Evolution for {player_name}')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.xticks(tournament_ticks, [f'Tournament {i}' for i in tournament_ticks])
        plt.show()

if __name__ == "__main__":
    results = scrape_results(URL)
    save_results_to_json(results)
    plot_player_evolution()





### DEPRECATED ###
# def update_ratings_from_results(results):
#     # Store each player as a Player object in a dictionary
#     players = {}
    
#     for round_results in results:
#         for white, black, result in round_results:
#             if white not in players:
#                 players[white] = Player()
#             if black not in players:
#                 players[black] = Player()

#             if result == "1-0":
#                 white_result, black_result = 1, 0
#             elif result == "0-1":
#                 white_result, black_result = 0, 1
#             else:
#                 white_result, black_result = 0.5, 0.5

#             players[white].update_player([players[black].rating], [players[black].rd], [white_result])
#             players[black].update_player([players[white].rating], [players[white].rd], [black_result])

#     # For plotting, if required
#     plot_all_ratings(players)

# def plot_all_ratings(players_dict):
#     for player, player_obj in players_dict.items():
#         plt.plot(player_obj.get_ratings_history(), label=player, marker='o')

#     plt.xlabel('Match Number')
#     plt.ylabel('Rating')
#     plt.title('Rating Evolution over Time')
#     plt.legend()
#     plt.grid(True)
#     plt.show()