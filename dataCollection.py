import requests
import os
from bs4 import BeautifulSoup
import time
import csv
"""
WMMDS web page structure:
    All sound cut page (dropdown)
        Species recording page (by year of recording)
            recording audio download links
    

WHOI recording file name structure: 
    AABBBCCC
    AA: 2 digit number (base 10) corresponding to year of recording
    BBB: 3-digit number (base 10) corresponding to species identification code
    CCC: 3-digit number (base 26) corresponding to id of recording (starting from 1)
"""
def downloadFile(url, saveDir):
    """
    Downloads a given url file to a save path
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        filename = os.path.basename(url)
        savePath = os.path.join(saveDir, filename)
        with open(savePath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded '{url}' to '{savePath}'")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading '{url}': {e}")



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

def downloadAllFromSpecies(baseURL: str, savePath: str, timedelay: int = 2)->None:
    """
    Given a base URL for a species, download all sound files from that species to the designated path directory 
    (relative to the working directory). Impose a time delay to prevent throttling or a site crash
    """
    recordingURLs = getSpeciesURLs(baseURL)
    downloadURLs = []
    for url in recordingURLs:
        downloadURLs.extend(getSoundURLsFromURL(url))
    for fileURL in downloadURLs:
        downloadFile(fileURL, savePath)
        time.sleep(timedelay)

def downloadDataset(saveDir: str)->None:
    """
    Download the entire Watkins Marine Mammal Dataset
    """
    with open("WMMDSmetadata.csv") as file:
        metadata = csv.reader(file)
        next(metadata) #skip header
        urlHeader = "https://cis.whoi.edu/science/B/whalesounds/"
        for species in metadata:
            baseURL = urlHeader + species[1]
            speciesName = species[2][1:-1] #strip quotes from csv
            isDownloaded = (species[3] == "TRUE")
            if isDownloaded: #skip downloading files that have already been downloaded
                continue
            speciesSaveDir = os.path.join(saveDir, speciesName)
            if not os.path.exists(speciesSaveDir): #create save directory if it does not exist
                os.mkdir(speciesSaveDir)
            downloadAllFromSpecies(baseURL, speciesSaveDir)

if __name__ == "__main__":
    #baseURL = "https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm?SP=CC3A&YR=-1"
    #saveDir = os.path.join(os.getcwd(), "RawSoundData")
    #downloadAllFromSpecies(baseURL, saveDir)
    #https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm?SP=CC3A&YR=63
    #getSpeciesURLs("https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm?SP=BD15F&YR=-1")    
    #url = "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61025001.wav"
    #filename = os.path.basename(url)  # Extracts filename from URL
    #save_path = os.path.join(os.getcwd(), filename) # Saves in current directory
    #download_file(url, save_path)
    downloadDataset("RawSoundData")