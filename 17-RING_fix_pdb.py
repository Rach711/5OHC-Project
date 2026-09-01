import os
import glob

def process_pdb(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_file)
    output_file = os.path.join(output_dir, f"clean_{base_name}")

    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Change "ATOM  " to "HETATM" for any line containing "DO"
            if "DO" in line and line.startswith("ATOM  "):
                line = line.replace("ATOM  ", "HETATM", 1)
            outfile.write(line)

    print(f"Success: {base_name} -> {output_file}")

if __name__ == "__main__":
    # This script lives in .../Scripts/, and RING/ is a sibling folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ring_dir = os.path.join(script_dir, "..", "RING")

    # Find every *_RING.pdb file inside each structure subfolder,
    # e.g. RING/APC_637_hot/APC_637_1_RING.pdb
    pdb_files = glob.glob(os.path.join(ring_dir, "*", "*_RING.pdb"))

    if not pdb_files:
        print(f"No PDB files found under {ring_dir}")

    for pdb_file in pdb_files:
        # Write the cleaned file back into the same structure subfolder
        output_dir = os.path.dirname(pdb_file)
        process_pdb(pdb_file, output_dir)