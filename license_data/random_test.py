import json
import random


# Function to load JSON data from a file
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


# Function to save JSON data to a file
def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


# Load data from the input JSON file
input_filename = 'C:/Users/kumar/PycharmProjects/firstIdea/license_data/quest_ans.json'
data = load_json(input_filename)

# Shuffle the list
random.shuffle(data)
print(len(data))
# Save the shuffled data to a new JSON file
output_filename = 'shuffled_questions.json'  # Replace with your desired output file name
save_json(data, output_filename)

# print(f"Shuffled questions and answers have been saved to {output_filename}")
