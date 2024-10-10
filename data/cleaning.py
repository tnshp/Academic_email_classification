import json 
import os

path = 'data/mails'

file_list = os.listdir(path)
data = []
for f in file_list:
    with open(os.path.join(path, f), 'r') as json_file:
        data += json.load(json_file)

# with open(os.path.join(path, file_list[0]), 'r') as json_file:
#     data = json.load(json_file)

unique_labels = set(item['label'] for item in data)


# Convert the set back to a list (optional, for easier handling)
unique_labels = list(unique_labels)
print(unique_labels)

label_2_logit = {'Research Queries': 2, 'Research Queries Email': 2, 'Research Query': 2, 'Research': 2, 'Research Query Email': 2,
 'Sensitive': 0,  'Sensitive Email': 0,
 'General Query': 1,'General Query Email': 1, 'General': 1
}

for f in file_list:
    with open(os.path.join(path, f), 'r') as json_file:
        data = json.load(json_file)
        for item in data:
            item['label'] = label_2_logit[item['label']]

        with open(os.path.join(path, f), 'w') as json_file:
            json.dump(data, json_file, indent=4)

