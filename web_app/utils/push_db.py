import json 
import requests 
import argparse

def load_json(filepath):
    # Read from a JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# choose from push piece and recordings

def main(db="pieces"):
    json_path = f"{db}_db.json"
    server_url = "http://localhost:3001"
    
    data = load_json(json_path)
    
    for data_point in data:
        req_url = f"{server_url}/{db}"
        response = requests.post(req_url, json=data_point)
        print(response.json())
        
if __name__ == "__main__":
    # parse -m argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="choose from pieces or recordings")
    args = parser.parse_args()
    
    main(args.mode)