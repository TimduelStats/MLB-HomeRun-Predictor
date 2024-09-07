import pitcher_starter_fetcher
import statcast_fetcher
import stats_fetcher
from datetime import datetime, timedelta


# Get 15 date before today's date
today = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.strptime(today, '%Y-%m-%d') -
              timedelta(days=15)).strftime('%Y-%m-%d')

# Get today's hard hit data
statcast_fetcher.FangraphsScraper.get_batter_fb_for_date(
    start_date, file_path="today_hh_data.csv")
# Get today's fb data
stats_fetcher.FangraphsScraper.get_batter_fb_for_date(
    start_date, file_path="today_fb_data.csv")
# Get today's pitcher matchup data
pitcher_starter_fetcher.PitcherScraperAPI.scrape_data(str(today))
