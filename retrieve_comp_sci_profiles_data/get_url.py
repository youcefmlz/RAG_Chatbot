import requests
from bs4 import BeautifulSoup
import json
import time


firstPageURL = 'https://www.cs.manchester.ac.uk/about/people/academic-and-research-staff/'
# Base URL for the profiles page
base_url = 'https://research.manchester.ac.uk/en/persons/?page={}'

# List to hold all extracted profile data
all_profile_data = []


# print(f"Scraping page {page_number}...")
# url = firstPageURL.format(page_number)
response = requests.get(firstPageURL)
response.raise_for_status()  # Check if the request was successful

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')
# print(f'soup : {soup}')
# Find the profile links
profiles = soup.find_all('li', class_=['tabrowwhite','tabrowgrey']) 


for profile in profiles:

    divv = profile.find('div' , class_ = 'tabCol_30')

    a_tag = divv.find('a')
    if a_tag:
        # Get the name and the href if it's available
        name = a_tag.get_text(strip=True)
        profile_url = a_tag['href']
    else:
        # If no <a> tag, just get the name from the text
        name = divv.get_text(strip=True)
        profile_url = None  # No URL available
    
    all_profile_data.append({
        'Name': name,
        'Profile URL': profile_url
    })

time.sleep(1)  # Delay to avoid overwhelming the server

# Save all profile data to a JSON file
with open('comp_sci_profiles.json', 'w') as f:
    json.dump(all_profile_data, f, indent=4)

print("All profile data saved to profiles.json")
