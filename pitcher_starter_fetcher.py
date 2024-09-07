import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from collections import defaultdict

pitcher_hand_map = {'Bailey Falter': 'LHP', 'Cristopher Sánchez': 'LHP', 'Pablo López': 'RHP', 'Tarik Skubal': 'LHP', 'Keaton Winn': 'RHP', 'Jacob Waguespack': 'RHP', 'Freddy Peralta': 'RHP', 'Tyler Wells': 'RHP', 'Ryan Feltner': 'RHP', 'Kevin Gausman': 'RHP', 'Michael Wacha': 'RHP', 'Luis Severino': 'RHP', 'Max Fried': 'LHP', 'Trevor Rogers': 'LHP', 'Reid Detmers': 'LHP', 'Tanner Houck': 'RHP', 'Andrew Abbott': 'LHP', 'Chris Flexen': 'RHP', 'Dane Dunning': 'RHP', 'J.P. France': 'RHP', 'Jordan Wicks': 'LHP', 'Bryce Miller': 'RHP', 'Jake Irvin': 'RHP', 'Paul Blackburn': 'RHP', 'Steven Matz': 'LHP', 'Brandon Pfaadt': 'RHP', 'Michael King': 'RHP', 'Yoshinobu Yamamoto': 'RHP', 'Clarke Schmidt': 'RHP', 'Carlos Carrasco': 'RHP', 'Joe Ryan': 'RHP', 'Kenta Maeda': 'RHP', 'Simeon Woods Richardson': 'RHP', 'Matt Manning': 'RHP', 'Alec Marsh': 'RHP', 'Sean Manaea': 'LHP', 'Nick Lodolo': 'LHP', 'Garrett Crochet': 'LHP', 'Dakota Hudson': 'RHP', 'Yariel Rodríguez': 'RHP', 'Marco Gonzales': 'LHP', 'Spencer Turnbull': 'RHP', 'Andrew Heaney': 'LHP', 'Ronel Blanco': 'RHP', 'DL Hall': 'LHP', 'Dean Kremer': 'RHP', 'MacKenzie Gore': 'LHP', 'Joe Boyle': 'RHP', 'Logan Webb': 'RHP', 'Ryan Pepiot': 'RHP', 'Chris Sale': 'LHP', 'Max Meyer': 'RHP', 'Griffin Canning': 'RHP', 'Cooper Criswell': 'RHP', 'Cody Poteet': 'RHP', 'Triston McKenzie': 'RHP', 'Kyle Gibson': 'RHP', 'Ryne Nelson': 'RHP', 'Matt Waldron': 'RHP', 'Gavin Stone': 'RHP', 'Shota Imanaga': 'LHP', 'Emerson Hancock': 'RHP', 'Mitch Keller': 'RHP', 'Zack Wheeler': 'RHP', 'Tyler Anderson': 'LHP', 'Brayan Bello': 'RHP', 'Colin Rea': 'RHP', 'Corbin Burnes': 'RHP', 'Kyle Freeland': 'LHP', 'José Berríos': 'RHP', 'Blake Snell': 'LHP', 'Shawn Armstrong': 'RHP', 'Cole Ragans': 'LHP', 'José Buttó': 'RHP', 'Charlie Morton': 'RHP', 'Jesús Luzardo': 'LHP', 'Bailey Ober': 'RHP', 'Jack Flaherty': 'RHP', 'Nestor Cortes': 'LHP', 'Logan Allen': 'LHP', 'Nathan Eovaldi': 'RHP', 'Cristian Javier': 'RHP', 'Graham Ashcraft': 'RHP', 'Michael Soroka': 'RHP', 'Trevor Williams': 'RHP', 'Alex Wood': 'LHP', 'Javier Assad': 'RHP', 'Luis Castillo': 'RHP', 'Miles Mikolas': 'RHP', 'Zac Gallen': 'RHP', 'Yu Darvish': 'RHP', 'James Paxton': 'LHP', 'Xzavion Curry': 'RHP', 'Kutter Crawford': 'RHP', 'Louie Varland': 'RHP', 'Cole Irvin': 'LHP', 'Cal Quantrill': 'RHP', 'Aaron Nola': 'RHP', 'Kyle Harrison': 'LHP', 'Edward Cabrera': 'RHP', 'Michael Lorenzen': 'RHP', 'Reese Olson': 'RHP', 'Patrick Sandoval': 'LHP', 'Zach Eflin': 'RHP', 'Luis Gil': 'RHP', 'Chris Bassitt': 'RHP', 'Martín Pérez': 'LHP', 'Adrian Houser': 'RHP', 'Joe Musgrove': 'RHP', 'Joe Ross': 'RHP', 'Seth Lugo': 'RHP', 'Nick Nastrini': 'RHP', 'Darius Vines': 'RHP', 'Spencer Arrighetti': 'RHP', 'Sonny Gray': 'RHP', 'Ross Stripling': 'RHP', 'Ben Brown': 'RHP', 'Merrill Kelly': 'RHP', 'Frankie Montas': 'RHP', 'George Kirby': 'RHP', 'Mitchell Parker': 'LHP', 'Tyler Glasnow': 'RHP', 'Jon Gray': 'RHP', 'Casey Mize': 'RHP', 'Chris Paddack': 'RHP', 'Grayson Rodriguez': 'RHP', 'Austin Gomber': 'LHP', 'Ranger Suárez': 'LHP', 'Jordan Hicks': 'RHP', 'Ryan Weathers': 'LHP', 'José Soriano': 'RHP', 'Aaron Civale': 'RHP', 'Carlos Rodón': 'LHP', 'Yusei Kikuchi': 'LHP', 'Jared Jones': 'RHP', 'Jose Quintana': 'LHP', 'Tanner Bibee': 'RHP', 'Garrett Whitlock': 'RHP', 'Dylan Cease': 'RHP', 'Wade Miley': 'LHP', 'Reynaldo López': 'RHP', 'Hunter Brown': 'RHP', 'Hunter Greene': 'RHP', 'Logan Gilbert': 'RHP', 'Lance Lynn': 'RHP', 'JP Sears': 'LHP', 'Kyle Hendricks': 'RHP', 'Tommy Henry': 'LHP', 'Patrick Corbin': 'LHP', 'Kyle Hurt': 'RHP', 'Brady Singer': 'RHP', 'Jonathan Cannon': 'RHP', 'Albert Suárez': 'RHP', 'Bryse Wilson': 'RHP', 'Erick Fedde': 'RHP', 'Marcus Stroman': 'RHP', 'Landon Knack': 'RHP', 'Zack Littell': 'RHP', 'Ben Lively': 'RHP', 'Jack Leiter': 'RHP', 'Brennan Bernardino': 'LHP', 'A.J. Puk': 'LHP', 'Jameson Taillon': 'RHP', 'Quinn Priester': 'RHP', 'Justin Verlander': 'RHP', 'Tyler Alexander': 'LHP', 'Jordan Montgomery': 'LHP', 'Roddery Muñoz': 'RHP',
                    'Randy Vásquez': 'RHP', 'Josh Winckowski': 'RHP', 'Slade Cecconi': 'RHP', 'Peter Lambert': 'RHP', 'Bryce Elder': 'RHP', 'Tobias Myers': 'RHP', 'Ryan Walker': 'RHP', 'Josh Fleming': 'LHP', 'Sixto Sánchez': 'RHP', 'Ty Blach': 'LHP', 'Chase Anderson': 'RHP', 'Nick Martinez': 'RHP', 'Anthony Maldonado': 'RHP', 'Michael Grove': 'RHP', 'Framber Valdez': 'LHP', 'Taijuan Walker': 'RHP', 'Hayden Wesneski': 'RHP', 'Jonathan Bowlan': 'RHP', 'Brandon Hughes': 'LHP', 'Erik Miller': 'LHP', 'Kyle Bradish': 'RHP', 'Brad Keller': 'RHP', 'John Means': 'LHP', 'Christian Scott': 'LD', 'Alek Manoah': 'RHP', 'Daniel Lynch IV': 'LHP', 'Matthew Liberatore': 'LHP', 'Mason Black': 'RHP', 'Mike Clevinger': 'RHP', 'Justin Steele': 'LHP', 'Walker Buehler': 'RHP', 'José Ureña': 'RHP', 'Osvaldo Bido': 'RHP', 'Nick Pivetta': 'RHP', 'Taj Bradley': 'RHP', 'Robert Gasser': 'LHP', 'Bryan Woo': 'RHP', 'Paul Skenes': 'RHP', 'Joey Estes': 'RHP', 'Braxton Garrett': 'LHP', 'Joey Lucchesi': 'LHP', 'Aaron Brooks': 'RHP', 'Elieser Hernández': 'RHP', 'Brent Suter': 'LHP', 'Mitch Spence': 'RHP', 'Tylor Megill': 'RHP', 'Joe Mantiply': 'LHP', 'AJ Smith-Shawver': 'RHP', 'Ray Kerr': 'LHP', 'Jared Koenig': 'LHP', 'Gerson Garabito': 'RHP', 'Blake Walston': 'LHP', 'Anthony Molina': 'RHP', 'Jake Woodford': 'RHP', 'Keider Montero': 'RHP', 'Andre Pallante': 'RHP', 'David Peterson': 'LHP', 'Spencer Schwellenbach': 'RHP', 'Hogan Harris': 'LHP', 'Luis Medina': 'RHP', 'DJ Herz': 'LHP', 'Trevor Richards': 'RHP', 'Adam Mazur': 'RHP', 'Aaron Ashby': 'LHP', 'Nick Sandlin': 'RHP', 'Cade Povich': 'LHP', 'Spencer Howard': 'RHP', 'Carmen Mlodzinski': 'RHP', 'Hurston Waldrep': 'RHP', 'Zack Kelly': 'RHP', 'Bowden Francis': 'RHP', 'Scott McGough': 'RHP', 'Carlos Rodriguez': 'RHP', 'Drew Thorpe': 'RHP', 'Jhonathan Díaz': 'LHP', 'José Suarez': 'LHP', 'Dan Altavilla': 'RHP', 'Shaun Anderson': 'RHP', 'Ben Joyce': 'RHP', 'Carson Spiers': 'RHP', 'Zach Plesac': 'RHP', 'Yonny Chirinos': 'RHP', 'Gerrit Cole': 'RHP', 'Bobby Miller': 'RHP', 'Jake Bloss': 'RHP', 'Kyle Tyler': 'RHP', 'Max Scherzer': 'RHP', 'Hoby Milner': 'LHP', 'Randy Rodríguez': 'RHP', 'Tyler Holton': 'LHP', 'Luis L. Ortiz': 'RHP', 'Dallas Keuchel': 'LHP', 'Valente Bellozo': 'RHP', 'Roansy Contreras': 'RHP', 'Ryan Burr': 'RHP', 'Hayden Birdsong': 'RHP', 'David Festa': 'RHP', 'Chad Kuhl': 'RHP', 'Davis Daniel': 'RHP', 'Shawn Dubin': 'RHP', 'Spencer Bivens': 'RHP', 'Rob Zastryzny': 'LHP', 'Michael Mercado': 'RHP', 'Gavin Williams': 'RHP', 'Cristian Mena': 'RHP', 'Jared Shuster': 'LHP', 'Bryan Hoeing': 'RHP', 'Shane Baz': 'RHP', 'Alex Faedo': 'RHP', 'Tanner Gordon': 'RHP', 'Justin Wrobleski': 'LHP', 'Yilber Diaz': 'RHP', 'Gordon Graceffo': 'RHP', 'Anthony Banda': 'LHP', 'Jack Kochanowicz': 'RHP', 'Jackson Rutledge': 'RHP', 'Tyler Phillips': 'RHP', 'Orion Kerkering': 'RHP', 'Germán Márquez': 'RHP', 'Brent Honeywell': 'RHP', 'Beau Brieske': 'RHP', 'Carson Fulmer': 'RHP', 'River Ryan': 'RHP', 'Allan Winans': 'RHP', 'Steven Okert': 'LHP', 'Chayce McDermott': 'RHP', 'Robbie Ray': 'LHP', 'Clayton Kershaw': 'LHP', 'Kenny Rosenberg': 'LHP', 'Kodai Senga': 'RHP', 'Joey Cantillo': 'LHP', 'Kolby Allard': 'LHP', 'Grant Holmes': 'RHP', 'Tayler Scott': 'RHP', 'Will Warren': 'RHP', 'Jeffrey Springs': 'LHP', 'Tony Santillan': 'RHP', 'Paolo Espino': 'RHP', 'Michael McGreevy': 'RHP', 'Davis Martin': 'RHP', 'Cody Bradford': 'LHP', 'Hunter Bigge': 'RHP', 'Ky Bush': 'LHP', 'Tyler Mahle': 'RHP', 'Eduardo Rodriguez': 'LHP', 'Brenan Hanifee': 'RHP', 'Alex Cobb': 'RHP', 'Drew Rasmussen': 'RHP', 'Bradley Blalock': 'RHP', 'Matthew Boyd': 'LHP', 'Zebby Matthews': 'RHP', 'Emilio Pagán': 'RHP', 'Adam Oller': 'RHP', 'Julian Aguiar': 'RHP', 'Domingo Germán': 'RHP', 'Johnny Cueto': 'RHP', 'Brock Burke': 'LHP', 'Buck Farmer': 'RHP', 'Jacob Lopez': 'LHP', 'Ty Madden': 'RHP', 'Brant Hurter': 'LHP', 'Jakob Junis': 'RHP', 'Mason Englert': 'RHP', 'Fernando Cruz': 'RHP', 'Matt Foster': 'RHP', 'J.T. Ginn': 'RHP', 'Rhett Lowder': 'RHP', 'Samuel Aldegheri': 'LHP'}

team_abbreviation_map = {
    'New York Yankees': 'NYY',
    'New York Mets': 'NYM',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'Atlanta Braves': 'ATL',
    'Washington Nationals': 'WSN',
    'Miami Marlins': 'MIA',
    'St. Louis Cardinals': 'STL',
    'Chicago Cubs': 'CHC',
    'Cincinnati Reds': 'CIN',
    'Milwaukee Brewers': 'MIL',
    'Los Angeles Dodgers': 'LAD',
    'San Francisco Giants': 'SFG',
    'Arizona Diamondbacks': 'ARI',
    'San Diego Padres': 'SDP',
    'Colorado Rockies': 'COL',
    'Boston Red Sox': 'BOS',
    'Toronto Blue Jays': 'TOR',
    'Tampa Bay Rays': 'TBR',
    'Baltimore Orioles': 'BAL',
    'Chicago White Sox': 'CHW',
    'Cleveland Guardians': 'CLE',
    'Detroit Tigers': 'DET',
    'Kansas City Royals': 'KCR',
    'Minnesota Twins': 'MIN',
    'Houston Astros': 'HOU',
    'Los Angeles Angels': 'LAA',
    'Seattle Mariners': 'SEA',
    'Texas Rangers': 'TEX',
    'Oakland Athletics': 'OAK',
}

# Team counter to prevent fetch wrond data
team_counter = defaultdict(int)

class PitcherScraperAPI:

    @staticmethod
    def fetch_data(date_str):
        """
        Fetch data from the MLB API for a given date.
        """
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&leagueId=103,104&hydrate=probablePitcher&language=en"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data for {date_str}. Status code: {response.status_code}")
            return None

    @staticmethod
    def parse_data(json_data):
        """
        Parses the pitcher data from the API JSON response.
        Returns a list of dictionaries with team names, pitcher names, and handedness.
        """
        data = []
        matchup_id = 1

        for date_info in json_data.get('dates', []):
            for game in date_info.get('games', []):
                game_date = game['officialDate']
                team_away = game['teams']['away']['team']['name']
                team_home = game['teams']['home']['team']['name']

                team_away_abbr = team_abbreviation_map.get(team_away, None)
                team_home_abbr = team_abbreviation_map.get(team_home, None)

                if not team_away_abbr or not team_home_abbr:
                    continue

                # Prevent fetching duplicate team data
                team_counter[team_away_abbr] += 1
                team_counter[team_home_abbr] += 1
                if team_counter[team_away_abbr] > 1 or team_counter[team_home_abbr] > 1:
                    continue

                # Extract probable pitchers
                away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName')
                home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName')

                if away_pitcher and home_pitcher:
                    data.append({
                        'matchup_id': matchup_id,
                        'game_date': game_date,
                        'team_away': team_away_abbr,
                        'pitcher_away': away_pitcher,
                        'pitcher_away_hand': pitcher_hand_map.get(away_pitcher, 'Unknown'),
                        'team_home': team_home_abbr,
                        'pitcher_home': home_pitcher,
                        'pitcher_home_hand': pitcher_hand_map.get(home_pitcher, 'Unknown'),
                    })
                    matchup_id += 1

        return data

    @staticmethod
    def save_data(data, file_path="pitcher_matchup_data.csv"):
        """
        Save the data to a CSV file.
        Args:
            data (list): The data to save.
            file_path (str): The file path to save the data.
        """
        df = pd.DataFrame(data)  # Convert list of dictionaries to DataFrame
        # Check if file exists. If it does, don't write the header again.
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', index=False, header=False)
        else:
            df.to_csv(file_path, mode='a', index=False, header=True)

    @staticmethod
    def scrape_data(date_str, file_path="pitcher_matchup_data.csv"):
        """
        Scrape probable pitchers from the MLB API for the given date.
        """
        print(f"Fetching data for {date_str}...")
        json_data = PitcherScraperAPI.fetch_data(date_str)
        if json_data:
            pitcher_data = PitcherScraperAPI.parse_data(json_data)
            if pitcher_data:
                PitcherScraperAPI.save_data(pitcher_data, file_path)
                print(f"Data saved to {file_path}")
            else:
                print("No data available to save.")
        else:
            print(f"No data fetched for {date_str}.")

if __name__ == "__main__":
    scraper = PitcherScraperAPI()
    scraper.scrape_data("2024-09-07")