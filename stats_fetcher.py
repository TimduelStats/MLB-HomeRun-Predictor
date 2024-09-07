from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import json
from datetime import datetime, timedelta
import os


class FangraphsScraper:

    @staticmethod
    def fetch_data(url):
        # Initialize the Chrome options
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless=new")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
        # Overcome limited resource problems
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Disable GPU hardware acceleration
        chrome_options.add_argument("--disable-gpu")
        # Debugging options (optional)
        chrome_options.add_argument("--remote-debugging-pipe")
        chrome_options.add_argument(
            "--log-path=/tmp/chrome_debug.log")  # Log path (optional)

        # Initialize the Selenium WebDriver with the configured Chrome options
        driver = webdriver.Chrome(options=chrome_options)

        try:
            # Fetch the webpage
            driver.get(url)

            # Wait for the table to load on the page
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, ".fg-data-grid.table-type tbody tr"))
            )

            # Extract the HTML content of the page
            html_content = driver.page_source
        finally:
            # Ensure the driver quits to release resources
            driver.quit()

        return html_content

    @staticmethod
    def parse_data(html_content, data_type, date):
        """
        Parses the data from the Fangraphs website and adds the date.
        Args:
            html_content (str): The HTML content to parse.
            data_type (str): Type of data to extract (e.g., 'fb' for Fly Ball%).
            date (str): The date to add to the data (formatted as YYYY-MM-DD).
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        table_div = soup.find('div', {'class': 'fg-data-grid table-type'})
        rows = table_div.find('table').find_all('tr', class_=True)
        data = []
        batter_id = 1  # Start with batter_id 1, increment for each batter

        for row in rows:
            columns = row.find_all('td')
            batter_name = columns[1].text.strip()  # Get batter name
            team = columns[2].text.strip()  # Get team name
            fb = columns[18].text.strip()  # Get Fly Ball percentage (FB%)

            # Append the parsed data
            data.append({
                'batter': batter_name,
                'fb': fb,
                'team': team,
                'batter_id': batter_id,
                'date': date  # Add the date
            })
            batter_id += 1  # Increment batter_id for the next batter

        return data

    @staticmethod
    def save_data(data, file_path="fb_data.csv"):
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
    def generate_url(base_url, start_date):
        """
        Generate the URL with the specific date range for the 3/28 batter FB%.
        Args:
            base_url (str): The base URL to append the date range to.
            start_date (str): The specific date for the data.
        Returns:
            str: The generated URL.
        """
        # Set the start and end date as 3/28
        # Parse the start date string to a datetime object
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')

        # Set end date to 15 days after the start date
        end_date_obj = start_date_obj + timedelta(days=14)

        # Format the dates back to 'YYYY-MM-DD' for URL usage
        start_date_str = start_date_obj.strftime('%Y-%m-%d')
        end_date_str = end_date_obj.strftime('%Y-%m-%d')
        return f"{base_url}?pos=all&stats=bat&lg=all&qual=y&type=23&season=2024&month=1000&season1=2024&ind=0&startdate={start_date_str}&enddate={end_date_str}&team=0&pagenum=1&pageitems=2000000000"

    @staticmethod
    def get_batter_fb_for_date(date, file_path="fb_data.csv"):
        """
        Fetch Fly Ball% for batters on a specific date and add the date to the data.
        Args:
            date (str): The date in the format 'YYYY-MM-DD'.
            file_path (str): The file path to save the data.
        """
        base_url = "https://www.fangraphs.com/leaders/major-league"
        url = FangraphsScraper.generate_url(base_url, date)
        print(url)

        # Calculate the date 16 days after the input date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        final_date = (date_obj + timedelta(days=15)).strftime('%Y-%m-%d')

        # Fetch the data
        html_content = FangraphsScraper.fetch_data(url)
        if html_content:
            # Parse and process the Fly Ball% data
            parsed_data = FangraphsScraper.parse_data(
                html_content, data_type="fb", date=final_date)
            FangraphsScraper.save_data(parsed_data, file_path)
            return parsed_data

# Function to iterate over a date range and scrape the data


def scrape_date_range(start_date_str, end_date_str):
    """
    Iterate over a date range from start_date to end_date in increments of 16 days
    and scrape the Fly Ball% data.
    Args:
        start_date_str (str): The start date in 'YYYY-MM-DD' format.
        end_date_str (str): The end date in 'YYYY-MM-DD' format.
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    while start_date <= end_date:
        scraper = FangraphsScraper()
        scraper.get_batter_fb_for_date(start_date.strftime('%Y-%m-%d'))
        start_date += timedelta(days=1)


if __name__ == "__main__":
    # Example usage to run from 2024-03-28 to 2024-08-16
    scrape_date_range("2024-03-28", "2024-08-16")
