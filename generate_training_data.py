# -*- coding: utf-8 -*-
import json
import csv

# your class for the training
trainDict = {
        "A": 0,    "C": 1,    "D": 2,    "E": 3,    "F": 4,
        "I": 5,    "H": 6,
    }
# trainDict = {
#         "A": 0,    "B": 1,    "C": 2,    "D": 3,    "E": 4,
#         "F": 5,    "G": 6,    "H": 7,
#     }

def main():
    # Read problemsheet
    with open('./data/pattern.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # write TSV files
    with open('./data/traindata.tsv', 'w', encoding='utf-8', newline='') as f:
        for i in json_data:
            pattern = i["problem"]
            id = i["id"]
            tw = csv.writer(f, delimiter='\t')
            tw.writerow([pattern, trainDict[id[0]]])



if __name__ == '__main__':
    main()