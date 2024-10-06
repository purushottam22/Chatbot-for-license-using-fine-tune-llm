import json


# Function to load JSON data from a file
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


data = load_json("quest_ans.json")
for i in range(len(data)):
    print(data[i]["question"])
