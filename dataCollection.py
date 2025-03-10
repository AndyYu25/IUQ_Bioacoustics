import requests
import os
from bs4 import BeautifulSoup
import time
"""
WHOI file name structure: 
    AABBBCCC
    AA: 2 digit number (base 10) corresponding to year of recording
    BBB: 3-digit number (base 10) corresponding to species identification code
    CCC: 3-digit number (base 26) corresponding to id of recording (starting from 1)
"""
def download_file(url, save_path):
    """
    Downloads a given url file to a save path
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded '{url}' to '{save_path}'")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading '{url}': {e}")

#url = "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61025001.wav"
#filename = os.path.basename(url)  # Extracts filename from URL
#save_path = os.path.join(os.getcwd(), filename) # Saves in current directory
#download_file(url, save_path)

def getSpeciesURLs(baseURL: str)->list[str]:
    """
    Given the base url of the form https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm?SP=AAAA&YR=-1
    where AAAA is the four digit code corresponding to a given mammal, navigate through the WMMSD to
    obtain all urls for each recording year of a given species.
    """
    baseWebsite = requests.get(baseURL) #get the "base" URL
    baseHTML = BeautifulSoup(baseWebsite.content, 'html.parser')
    yearDropdown = baseHTML.find(id = 'pickYear')
    #get all option elements under the year dropdown and extract their text to obtain all recording years
    options = yearDropdown.find_all('option') if yearDropdown else []
    #remove 1st element b/c it's the selected value and not a real option
    yearList = [option.get_text(strip=True) for option in options][1:]
    #create and return all the suburls from which the download urls will be obtained
    recordingURLs = []
    for year in yearList:
        recordingURL = baseURL[:-2] + year[2:]
        recordingURLs.append(recordingURL)
    return recordingURLs

#getSpeciesURLs("https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm?SP=BD15F&YR=-1")    

def getSoundURLsFromURL(url: str)->list[str]:
    """
    Given a url from the Watkins Marine Mammal Sound Database of the following form:
    https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm?SP=AAAA&YR=YY
    where AAAA the four digit code corresponding to a given mammal and YY is the 
    last two digits of the year of the recordings, return all sound file urls from the data.
    """
    baseWebsite = requests.get(url) #get the "base" URL & extract html file
    baseHTML = BeautifulSoup(baseWebsite.content, 'html.parser')
    recordingTable = baseHTML.find('table')
    #find each download element within the table, which has a target attribute of '_blank'
    tableEntries = recordingTable.find_all(target='_blank')
    #get the href attribute from each download element are parse them into proper urls
    urlList = []
    baseAddress = 'https://cis.whoi.edu'
    for entry in tableEntries:
        urlList.append(baseAddress + entry['href'])
    return urlList

def downloadAllFromSpecies(baseURL)

#https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm?SP=CC3A&YR=63