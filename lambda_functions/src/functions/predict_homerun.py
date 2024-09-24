import os
import sys
# Add the parent directory to the system path to find utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.s3_uploader import download_from_s3, upload_to_s3, delete_from_s3
import utils.player_map as player_map
import pandas as pd
import joblib
# Constants
BUCKET_NAME = 'timjimmymlbdata'
MODEL_FILE = 'xgb_model1.pkl'
PREDICTION_OUTPUT_FILE = 'predicted_homeruns2.csv'


def main(event, context):
    # Delete file from S3
    delete_from_s3(BUCKET_NAME, PREDICTION_OUTPUT_FILE)

    # Log the start of the Lambda function
    print("Home run prediction Lambda function started")

    # Download today's data from S3
    download_from_s3(BUCKET_NAME, 'today_iso_data.csv', '/tmp/today_iso_data.csv')
    download_from_s3(BUCKET_NAME, 'today_hh_data.csv', '/tmp/today_hh_data.csv')
    download_from_s3(BUCKET_NAME, 'today_pitcher_data.csv', '/tmp/pitcher_matchup_data.csv')

    # Load data into dataframes
    iso_data = pd.read_csv('/tmp/today_iso_data.csv')
    hh_data = pd.read_csv('/tmp/today_hh_data.csv')
    pitcher_matchup_data = pd.read_csv('/tmp/pitcher_matchup_data.csv')

    # Merge iso_data and hh_data on 'batter' and 'date'
    merged_data = pd.merge(iso_data, hh_data, on=['batter', 'date'], how='inner')

    # Merge pitcher's data based on date and game_date
    merged_data = pd.merge(merged_data, pitcher_matchup_data, left_on='date', right_on='game_date', how='left')

    # Filter rows where team matches either team_home or team_away
    merged_data = merged_data[
        (merged_data['team'] == merged_data['team_home']) |
        (merged_data['team'] == merged_data['team_away'])
    ]

    # Function to map opponent's pitcher and their hand
    def map_opponent(row):
        if row['team'] == row['team_home']:
            return row['pitcher_away'], row['pitcher_away_hand']
        else:
            return row['pitcher_home'], row['pitcher_home_hand']

    # Apply the function to determine the opponent's pitcher and hand
    merged_data[['pitcher', 'pitcher_hand']] = merged_data.apply(
        lambda row: pd.Series(map_opponent(row)), axis=1)

    # Drop unnecessary columns
    merged_data = merged_data.drop(columns=['game_date', 'team_away', 'team_home',
                                            'pitcher_away_hand', 'pitcher_away', 'pitcher_home', 'pitcher_home_hand',
                                            'matchup_id'])

    merged_data.drop(columns=['date', 'batter_id'], inplace=True)

    # Create batter and pitcher IDs using player_map
    merged_data['batter_id'] = merged_data['batter'].map(player_map.batter).fillna(-1).astype(int)
    merged_data['pitcher_id'] = merged_data['pitcher'].map(player_map.pitcher).fillna(-1).astype(int)

    # Filter out rows where batter_id or pitcher_id is -1
    merged_data = merged_data[merged_data['batter_id'] != -1]
    merged_data = merged_data[merged_data['pitcher_id'] != -1]

    # Convert pitcher_hand to binary (RHP = 1, LHP = 0)
    merged_data['pitcher_hand'] = merged_data['pitcher_hand'].apply(lambda x: 1 if x == 'RHP' else 0)

    # Convert iso and hh to binary
    merged_data['iso'] = merged_data['iso'].apply(lambda x: 1 if x >= 100 else 0)
    merged_data['hard_hit'] = merged_data['hard_hit'].apply(lambda x: 1 if x >= 10 else 0)

    # Drop unnecessary columns
    merged_data.drop(columns=['batter', 'team', 'pitcher'], inplace=True)

    # Load the XGBoost model
    model_path = f'/tmp/{MODEL_FILE}'
    download_from_s3(BUCKET_NAME, MODEL_FILE, model_path)
    xgb_model_loaded = joblib.load(model_path)

    # Define the features for prediction
    features = ['iso', 'hard_hit', 'pitcher_hand', 'batter_id', 'pitcher_id']
    batter = {1: 'Yandy Díaz', 2: 'Leody Taveras', 3: 'Taylor Ward', 4: 'Kyle Tucker', 5: 'Nico Hoerner', 6: 'Brice Turang', 7: 'Brandon Marsh', 8: 'Oneil Cruz', 9: 'Edouard Julien', 10: 'Anthony Rendon', 11: 'Harold Ramírez', 12: 'Steven Kwan', 13: 'Ceddanne Rafaela', 14: 'Bo Bichette', 15: 'Alec Bohm', 16: 'Mike Trout', 17: 'Pete Alonso', 18: 'Jared Triolo', 19: 'Tyler Freeman', 20: 'Josh Naylor', 21: 'Gleyber Torres', 22: 'Eugenio Suárez', 23: 'George Springer', 24: 'Bryan De La Cruz', 25: 'Joey Gallo', 26: 'Yoán Moncada', 27: 'Luis Arraez', 28: 'Alex Verdugo', 29: 'Jarren Duran', 30: 'Ryan McMahon', 31: 'Brendan Rodgers', 32: 'Ha-Seong Kim', 33: 'Jackson Chourio', 34: 'Bryson Stott', 35: "Tyler O'Neill", 36: 'Nelson Velázquez', 37: 'Ketel Marte', 38: 'Bryan Reynolds', 39: 'Corbin Carroll', 40: 'Alejandro Kirk', 41: 'Jack Suwinski', 42: 'Teoscar Hernández', 43: 'Matt Chapman', 44: "Ke'Bryan Hayes", 45: 'Bobby Witt Jr.', 46: 'Andrew Benintendi', 47: 'Julio Rodríguez', 48: 'Nick Castellanos', 49: 'William Contreras', 50: 'Seth Brown', 51: 'Jared Walsh', 52: 'Jonah Heim', 53: 'Ryan Mountcastle', 54: 'Willy Adames', 55: 'Connor Joe', 56: 'Ian Happ', 57: 'Alex Bregman', 58: 'Alec Burleson', 59: 'Trey Lipscomb', 60: 'Manny Machado', 61: 'Lane Thomas', 62: 'J.T. Realmuto', 63: 'Brett Baty', 64: 'Randy Arozarena', 65: 'Freddie Freeman', 66: 'Jake Cronenworth', 67: 'José Ramírez', 68: 'Mookie Betts', 69: 'Sal Frelick', 70: 'Aaron Judge', 71: 'Zack Gelof', 72: 'Lourdes Gurriel Jr.', 73: 'Jorge Soler', 74: 'Andrés Giménez', 75: 'Jeimer Candelario', 76: 'Josh Bell', 77: 'Giancarlo Stanton', 78: 'Enmanuel Valdez', 79: 'Lawrence Butler', 80: 'Zach Neto', 81: 'Michael Busch', 82: 'Jackson Merrill', 83: 'Anthony Santander', 84: 'J.P. Crawford', 85: 'Juan Soto', 86: 'Yordan Alvarez', 87: 'Shohei Ohtani', 88: 'Adolis García', 89: 'Daulton Varsho', 90: 'MJ Melendez', 91: 'Will Brennan', 92: 'Will Smith', 93: 'Brendan Donovan', 94: 'Marcus Semien', 95: 'Victor Scott II', 96: 'Mark Canha', 97: 'Henry Davis', 98: 'Adley Rutschman', 99: 'Cody Bellinger', 100: 'Anthony Volpe', 101: 'Seiya Suzuki', 102: 'Riley Greene', 103: 'Christian Yelich', 104: 'Brenton Doyle', 105: 'Carlos Correa', 106: 'Orlando Arcia', 107: 'Jeremy Peña', 108: 'Spencer Steer', 109: 'Francisco Alvarez', 110: 'Jung Hoo Lee', 111: 'Rafael Devers', 112: 'Nolan Arenado', 113: 'Maikel Garcia', 114: 'Paul Goldschmidt', 115: 'Nolan Jones', 116: 'CJ Abrams', 117: 'Bryce Harper', 118: 'Austin Riley', 119: 'Anthony Rizzo', 120: 'Vladimir Guerrero Jr.', 121: 'Charlie Blackmon', 122: 'Ryan Noda', 123: 'Ronald Acuña Jr.', 124: 'Matt Olson', 125: 'Tim Anderson', 126: 'Ezequiel Tovar', 127: 'Byron Buxton', 128: 'Alex Kirilloff', 129: 'José Caballero', 130: 'JJ Bleday', 131: 'Starling Marte', 132: 'Jonathan India', 133: 'Chas McCormick', 134: 'Evan Carter', 135: 'Jordan Westburg', 136: 'Marcell Ozuna', 137: 'Michael Conforto', 138: 'Thairo Estrada', 139: 'Dansby Swanson', 140: 'Fernando Tatis Jr.', 141: 'Oliver Dunn', 142: 'Gunnar Henderson', 143: 'Justin Turner', 144: 'Andrew Vaughn', 145: 'Colt Keith', 146: 'Jeff McNeil', 147: 'Gabriel Moreno', 148: 'Yainer Diaz', 149: 'Elly De La Cruz', 150: 'Jordan Walker', 151: 'Brandon Nimmo', 152: 'Cedric Mullins', 153: 'Christopher Morel', 154: 'Javier Báez', 155: 'Triston Casas', 156: 'Vinnie Pasquantino', 157: 'Jake Burger', 158: 'Corey Seager', 159: 'J.D. Davis', 160: 'Kris Bryant', 161: 'Will Benson', 162: 'Christian Encarnacion-St...', 163: 'Mitch Haniger', 164: 'Willi Castro', 165: 'Masataka Yoshida', 166: 'Jurickson Profar', 167: 'Wyatt Langford', 168: 'Nick Ahmed', 169: 'Christian Walker', 170: 'Michael Harris II', 171: 'Nolan Gorman', 172: 'Brandon Drury', 173: 'Jorge Polanco', 174: 'Jose Siri', 175: 'Spencer Torkelson', 176: 'Brayan Rocchio', 177: 'Jazz Chisholm Jr.', 178: 'Francisco Lindor', 179: 'Max Muncy', 180: 'Salvador Perez', 181: 'Isaac Paredes', 182: 'Rhys Hoskins', 183: "Logan O'Hoppe", 184: 'Jose Altuve', 185: 'Joey Meneses', 186: 'Nolan Schanuel', 187: 'Xander Bogaerts', 188: 'Jesse Winker', 189: 'Trea Turner', 190: 'Kyle Schwarber', 191: 'Ozzie Albies', 192: 'Carlos Santana', 193: 'Gio Urshela', 194: 'Shea Langeliers', 195: 'Bo Naylor', 196: 'Ty France', 197: 'James Outman', 198: 'Masyn Winn', 199: 'Mitch Garver', 200: 'Cavan Biggio', 201: 'Rowdy Tellez', 202: 'Abraham Toro', 203: 'Isiah Kiner-Falefa', 204: 'Josh Smith', 205: 'Gavin Sheets', 206: 'Elias Díaz', 207: 'Cal Raleigh', 208: 'Harrison Bader', 209: 'Colton Cowser', 210: 'Amed Rosario', 211: 'Robbie Grossman', 212: 'Luis Campusano', 213: 'Dominic Fletcher', 214: 'Elehuris Montero', 215: 'Kerry Carpenter', 216: "Ryan O'Hearn", 217: 'Ryan Jeffers', 218: 'Willson Contreras', 219: 'Blake Perkins', 220: 'Luis Rengifo', 221: 'Luis García Jr.', 222: 'Nicky Lopez', 223: 'Tyler Nevin', 224: 'Curtis Mead', 225: 'Austin Martin', 226: 'Mike Tauchman', 227: 'Mauricio Dubón', 228: 'Lars Nootbaar', 229: 'Wilyer Abreu', 230: 'Matt Vierling', 231: 'Wilmer Flores', 232: 'Oswaldo Cabrera', 233: 'Tyler Stephenson', 234: 'Riley Adams', 235: 'Emmanuel Rivera', 236: 'Andrew McCutchen', 237: 'Johan Rojas', 238: 'Eloy Jiménez', 239: 'Andy Pages', 240: 'Josh Rojas', 241: 'Connor Wong', 242: 'Joc Pederson', 243: 'Nick Senzel', 244: 'Santiago Espinal', 245: 'Brent Rooker', 246: 'Richie Palacios', 247: 'Nathaniel Lowe', 248: 'Danny Mendick', 249: 'Nick Martini', 250: 'Rob Refsnyder', 251: 'Wenceel Pérez', 252: 'Jo Adell', 253: 'Jesús Sánchez', 254: 'Dylan Moore', 255: 'Kyle Isbel', 256: 'Max Kepler', 257: 'Paul DeJong', 258: 'Jake McCarthy', 259: 'Michael Massey', 260: 'Jacob Young', 261: 'Tommy Pham', 262: 'Jon Singleton', 263: 'J.D. Martinez', 264: 'Vidal Bruján', 265: 'Pete Crow-Armstrong', 266: 'Davis Schneider', 267: 'Keibert Ruiz', 268: 'LaMonte Wade Jr.', 269: 'Danny Jansen', 270: 'Hunter Renfroe', 271: 'Brett Harris', 272: 'Jordan Beck', 273: 'Mike Yastrzemski', 274: 'Eddie Rosario', 275: 'Willie Calhoun', 276: 'Whit Merrifield', 277: 'Jose Miranda', 278: 'Jorge Mateo', 279: 'Vaughn Grissom', 280: 'Max Schuemann', 281: 'Dominic Smith', 282: 'Kevin Newman', 283: 'Iván Herrera', 284: 'Gavin Lux', 285: 'Nick Gordon', 286: 'Josh Lowe', 287: 'Joey Ortiz', 288: 'Jonny DeLuca', 289: 'Luke Raley', 290: 'Jake Meyers', 291: 'Mike Ford', 292: 'Heliot Ramos', 293: 'Jake Fraley', 294: 'Otto Lopez', 295: 'Kevin Pillar', 296: 'Nick Gonzales', 297: 'Luis Matos', 298: 'Zack Short', 299: 'Michael Siani', 300: 'Jake Bauers', 301: 'Jonathan Aranda', 302: 'David Fry', 303: 'David Hamilton', 304: 'Korey Lee', 305: 'Corey Julks', 306: 'Edmundo Sosa', 307: 'Brandon Lowe', 308: 'Adam Duvall', 309: 'Patrick Bailey', 310: 'Jarred Kelenic', 311: 'Mark Vientos', 312: 'Miguel Andujar', 313: 'Donovan Solano', 314: 'TJ Friedl', 315: 'DJ LeMahieu', 316: 'Ezequiel Duran', 317: 'Nick Loftin', 318: 'Lenyn Sosa', 319: 'Luis Robert Jr.', 320: 'Tyler Soderstrom', 321: 'Royce Lewis', 322: 'Justyn-Henry Malloy', 323: 'Jake Cave', 324: 'Michael Toglia', 325: 'Mickey Moniak', 326: 'Spencer Horwitz', 327: 'Brett Wisely', 328: 'Pedro Pagés', 329: 'Ramón Urías', 330: 'Manuel Margot', 331: 'Jason Heyward', 332: 'Daniel Schneemann', 333: 'Hunter Goodman', 334: 'Geraldo Perdomo', 335: 'Miguel Rojas', 336: 'Ben Rice', 337: 'Stuart Fairchild', 338: 'Tyrone Taylor', 339: 'Taylor Walls', 340: 'Trevor Larnach', 341: 'Noelvi Marte', 342: 'Trent Grisham', 343: 'James Wood', 344: 'Ernie Clement', 345: 'Chris Taylor', 346: 'Xavier Edwards', 347: 'Angel Martínez', 348: 'Brooks Lee', 349: 'Joshua Palacios', 350: 'Freddy Fermin', 351: 'Juan Yepez', 352: 'Miles Mastrobuoni', 353: 'Rece Hinds', 354: 'Nick Fortes', 355: 'Adam Frazier', 356: 'Matt Wallner', 357: 'Leo Jiménez', 358: 'David Peralta', 359: 'Austin Wells', 360: 'Garrett Mitchell', 361: 'Victor Robles', 362: 'Jose Iglesias', 363: 'Enrique Hernández', 364: 'Brooks Baldwin', 365: 'Romy Gonzalez', 366: 'Tyler Fitzgerald', 367: 'Joey Bart', 368: 'Jonah Bride', 369: 'Bligh Madris', 370: 'Alex Call', 371: 'Miguel Vargas', 372: 'Josh Jung', 373: 'Jackson Holliday', 374: 'Michael A. Taylor', 375: 'Joey Loperfido', 376: 'Kyle Stowers', 377: 'Nick Sogard', 378: 'Jerar Encarnacion', 379: 'Parker Meadows', 380: 'Derek Hill', 381: 'Zach McKinstry', 382: 'Dylan Carlson', 383: 'Ramón Laureano', 384: 'Adrian Del Castillo', 385: 'Jhonkensy Noel', 386: 'Junior Caminero', 387: 'Andrés Chaparro', 388: 'Jace Jung', 389: 'Grant McCray', 390: 'Addison Barger', 391: 'José Tena', 392: 'Connor Norby', 393: 'Miguel Amaya'}
    # Make predictions
    predictions = xgb_model_loaded.predict(merged_data[features])

    # Add predictions to the DataFrame
    merged_data['predicted_homerun'] = predictions
    merged_data['batter_id'] = merged_data['batter_id'].map(batter)

    # Filter rows where predicted_homerun is not 0
    filtered_data = merged_data[merged_data['predicted_homerun'] != 0]

    # Add team name based on batter name
    filtered_data['team'] = filtered_data['batter_id'].map(player_map.player_team)

    # Save the final results to CSV
    output_path = f'/tmp/{PREDICTION_OUTPUT_FILE}'
    filtered_data.to_csv(output_path, index=False)

    # Upload the CSV file back to S3
    upload_to_s3(output_path, BUCKET_NAME, PREDICTION_OUTPUT_FILE)

    print(f"Prediction results saved and uploaded to S3: {PREDICTION_OUTPUT_FILE}")

    return {
        'statusCode': 200,
        'body': 'Home run predictions completed successfully.'
    }

if __name__ == '__main__':
    main(None, None)