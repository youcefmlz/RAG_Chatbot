import requests
from bs4 import BeautifulSoup
import json
import os


def create_filename(name):
    # Remove characters that are invalid in file names
    return "".join([c for c in name if c.isalnum() or c in [' ', '_']]).strip().replace(" ", "_") + '.json'

# Function to scrape the details of a specific profile
def scrape_profile(url, name):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    
    
    
    #  --------
# Find the overview section
    overview_section = soup.find('div', class_='rendering_profileinformationportal')

    # Initialize a dictionary to hold the sections
    sections = {}

    if overview_section:
        # Define the section headers to look for
        section_headers = ["Biography", "Overview", "Research interests","Memberships of committees and professional bodies"]

        for header in section_headers:
            # Find the <h3> element with the specific header
            h3 = overview_section.find('h3', text=header)
            if h3:
                # Get the following sibling which is the <div> with class 'textblock'
                textblock = h3.find_next_sibling('div', class_='textblock')
                if textblock:
                    sections[header] = textblock.get_text()
                else:
                    sections[header] = None
            else:
                sections[header] = None

        # Print the results
        for section, text in sections.items():
            print(f"{section}: {text}\n")
    else:
        print("Overview section not found.")
        
        
    # --- Find academic qualifications 
    academic_qualifications = soup.find_all('div', class_= 'rendering_personeducation')
    qualifications = []
    if len(academic_qualifications) !=0:
        for qualif in academic_qualifications:
            qualifications.append(qualif.text)
    else:
        print("Academic qualifications section not found.") 
    
    profile_data = {
        'Name': name,
        'Profile URL': url,
        'Overview' : sections['Overview'],
        'Biography': sections['Biography'],
        'Research interests':sections['Research interests'],
        'Memberships of committees and professional bodies':sections['Memberships of committees and professional bodies'],
        'Academic qualifications': qualifications
    }
    
    return profile_data

# Read the profiles.json file
with open('comp_sci_profiles.json', 'r') as f:
    profiles = json.load(f)

# Directory to save the JSON files
if not os.path.exists('profile_data'):
    os.makedirs('profile_data')

# Iterate over each profile and scrape details
for profile in profiles:
    name = profile['Name']
    profile_url = profile['Profile URL']  # Use the complete URL directly from JSON

    
    try:
        # Scrape each profile and get the data
        profile_data = scrape_profile(profile_url, name)
        
        # Create a filename for each person based on their name
        filename = create_filename(name)
        filepath = os.path.join('profile_data', filename)
        
        # Save the profile data to a JSON file
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=4)
        
        print(f"Saved {name}'s profile data to {filename}")
    
    except Exception as e:
        print(f"Error scraping {name}: {e}")


