import requests
import os
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

# Example usage:
url = "https://cis.whoi.edu/science/B/whalesounds/WhaleSounds/61025001.wav"
filename = os.path.basename(url)  # Extracts filename from URL
save_path = os.path.join(os.getcwd(), filename) # Saves in current directory

download_file(url, save_path)