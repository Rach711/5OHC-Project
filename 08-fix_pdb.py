import sys
import os

def process_pdb(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_file)
    output_file = os.path.join(output_dir, f"clean_{base_name}")
    
    prev_chain = None
    clear_5_chain = None
    clear_5_res = None
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Clean up line endings immediately (removes hidden Windows \r characters)
            line = line.replace('\r', '').replace('\n', '')
            
            # Skip records handled automatically by pdb2gmx
            if line.startswith(("CONECT", "TER", "END")) or not line.strip():
                continue
                
            if line.startswith(("ATOM", "HETATM")):
                # 1. Normalize HETATM records to ATOM so they are treated as a continuous polymer
                if line.startswith("HETATM"):
                    line = "ATOM  " + line[6:]
                
                # Safely slice text strings even if trailing spaces are missing
                line = line.ljust(80)
                
                res_name = line[17:20].strip()
                chain_id = line[21]
                res_seq = line[22:26].strip()
                
                # 2. Handle chain transitions and explicitly mark the 5' terminal residue
                if prev_chain is not None and chain_id != prev_chain:
                    outfile.write("TER\n")
                    if res_name in ["DA", "DC", "DG", "DT", "A", "C", "G", "T"]:
                        clear_5_chain = chain_id
                        clear_5_res = res_seq
                elif prev_chain is None:
                    if res_name in ["DA", "DC", "DG", "DT", "A", "C", "G", "T"]:
                        clear_5_chain = chain_id
                        clear_5_res = res_seq
                
                # Always safely keep track of the chain context right away
                prev_chain = chain_id
                
                atom_name = line[12:16]
                res_name_full = line[17:20]
                a_clean = atom_name.strip()
                r_clean = res_name.strip()
                
                # 3. Apply changes specifically for the custom DO residue
                if r_clean == "DO":
                    if a_clean in ["OP1", "O1P"]: 
                        atom_name = " O1P"
                    elif a_clean in ["OP2", "O2P"]: 
                        atom_name = " O2P"
                    else:
                        atom_name = f"{a_clean:<4}" if len(a_clean) == 4 else f" {a_clean:<3}"
    
                    res_name_full = " DO"
                    
                # 4. Apply changes for standard Protein & DNA residues
                else:
                    if r_clean == "A": res_name_full = " DA"
                    elif r_clean == "C": res_name_full = " DC"
                    elif r_clean == "G": res_name_full = " DG"
                    elif r_clean == "T": res_name_full = " DT"
                    
                    # HARDENED AUTOMATED FIX: If this atom belongs to the identified 5' terminal 
                    # residue of a DNA chain, strip the phosphate capping group completely.
                    if chain_id == clear_5_chain and res_seq == clear_5_res:
                        if a_clean in ["P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"]:
                            continue  # Safely skips writing the atom
                    
                    # Standardize regular internal/3' backbone phosphate naming variants
                    if a_clean == "OP1": atom_name = " O1P"
                    elif a_clean == "OP2": atom_name = " O2P"
                    elif a_clean == "OP3": atom_name = " O3P"
                    
                    # Fix standard Thymine methyl naming anomaly
                    if res_name_full.strip() == "DT" and a_clean == "C7": 
                        atom_name = " C5M"
                
                # 5. RECONSTRUCT THE LINE COMPLETELY
                line = f"{line[:12]}{atom_name}{line[16:17]}{res_name_full}{line[20:]}"
            
            outfile.write(line + "\n")
        
        outfile.write("TER\nEND\n")
        
    print(f"Success: {base_name} -> {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("command line should be: python fix_pdb.py <path/to/files/*.pdb>")
        sys.exit(1)
        
    target_dir = "../input/clean"
    for pdb_file in sys.argv[1:]:
        process_pdb(pdb_file, target_dir)