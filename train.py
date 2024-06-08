import json

# Path to your JSONL file
file_path = "All_Beauty.json"


# Function to extract values of specific keys from JSON objects
def extract_values(json_line, keys):
    data = json.loads(json_line)
    return [data[key] for key in keys if key in data]


# Define the keys you want to extract values for
keys_to_extract = ["title"]

# Open the JSONL file and extract values for the specified keys
with open(file_path, "r") as file:
    line = file.readline()
    print(type(line))