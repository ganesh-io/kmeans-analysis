import numpy as np
import csv

def load_csv(path):
    data = []
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)