import pandas as pd
from pathlib import Path
import os

# # Absolute paths to Curves+ binaries and standard library
# CURVES_BIN="/mnt/c/Users/rache/downloads/curves+/Cur+"
# CANAL_BIN="/mnt/c/Users/rache/downloads/curves+/canal"
# LIB_PATH="/mnt/c/Users/rache/downloads/curves+/standard"

# BASE_DIR = Path("/../output")
# sequences = ["APC_637_hot", 
#              "APC_641_non", 
#              "APC_3335_non", 
#              "APC_3340_hot", 
#              "APC_4099_hot", 
#              "APC_4103_non", 
#              "APC_4343_non", 
#              "APC_4348_hot", 
#              "TP53_632_non", 
#              "TP53_637_hot", 
#              "TP53_844_hot", 
#              "TP53_849_non"]

# replicates = [1, 2, 3]

# with open("C:/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/output/APC_637_hot/1/curves/APC_637_1-1-2.lis") as f:
#     for line in f:
#         parts = line.split()   # breaks the line into a list of words, splitting on spaces
#         print(parts)

# first_word = parts[0]
# if first_word.endswith(")") and first_word[:-1].isdigit():
#     def parse_lis_file(filepath):
#     results = {}
#     with open(filepath) as f:
#         for line in f:
#             if "distribution" in line:
#                 break   # stop reading entirely — everything after this is a different table
#             parts = line.split()
#             if len(parts) < 4:
#                 continue
#             if not (parts[0].endswith(")") and parts[0][:-1].isdigit()):
#                 continue
#             var = parts[2]
#             aver = parts[3]
#             results[var] = aver
#     return results

def lis_parser(file):
    results = {}
    with open(file) as f:
        for line in f:
            if "distribution" in line:
                break 
            parts = line.split()
            if len(parts) < 4:
                continue
            if not (parts[0].endswith(")") and parts[0][:-1].isdigit()):
                continue
            var = parts[2]
            aver = float(parts[3])
            results[var] = aver # "tbend": "88.2"
    return pd.DataFrame(results.items(), columns=['Var', 'Aver'])

BASE_PATH = "C:/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/output"

for x in os.listdir(BASE_PATH):
    sequence = os.path.join(BASE_PATH, x)
    for num in os.listdir(sequence):
        print(os.path.join(sequence, num))
        data = lis_parser(file_path)

test = f"{BASE_PATH}/APC_637_hot/1/curves"
for z in os.listdir(test): # order list for 1,2 instead of 1,10
    
    if "APC_637_1-" in z:
        file_path = os.path.join(test, z)

        

