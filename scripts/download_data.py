"""
Script to download data files from external storage.
"""
import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, local_filename):
    """Download a file from a URL with progress bar."""
    if os.path.exists(local_filename):
        print(f"File already exists: {local_filename}")
        return local_filename
    
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(local_filename, 'wb') as f:
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(local_filename))
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
            pbar.close()
    
    return local_filename

def extract_zip(zip_file, extract_dir):
    """Extract a zip file to a directory."""
    if not os.path.exists(zip_file):
        print(f"Zip file not found: {zip_file}")
        return
    
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_file, 'r') as z:
        z.extractall(extract_dir)
    
    print(f"Extracted {zip_file} to {extract_dir}")

def main():
    """Main function to download and extract data files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Replace these URLs with your actual data URLs from cloud storage
    downloads = [
        {
            'url': 'https://your-storage-url.com/nwm_forecasts.zip',
            'local_path': os.path.join(base_dir, 'data', 'downloads', 'nwm_forecasts.zip'),
            'extract_dir': os.path.join(base_dir, 'data', 'raw'),
        },
        {
            'url': 'https://your-storage-url.com/usgs_observations.zip',
            'local_path': os.path.join(base_dir, 'data', 'downloads', 'usgs_observations.zip'),
            'extract_dir': os.path.join(base_dir, 'data', 'raw'),
        },
        {
            'url': 'https://your-storage-url.com/models.zip',
            'local_path': os.path.join(base_dir, 'data', 'downloads', 'models.zip'),
            'extract_dir': os.path.join(base_dir),
        }
    ]
    
    # Download and extract each file
    for item in downloads:
        print(f"Processing {os.path.basename(item['local_path'])}...")
        download_file(item['url'], item['local_path'])
        extract_zip(item['local_path'], item['extract_dir'])
    
    print("All data files downloaded and extracted successfully!")

if __name__ == "__main__":
    main()
