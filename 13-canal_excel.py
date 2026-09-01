import pandas as pd
from pathlib import Path

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
            results[var] = aver
    df = pd.DataFrame(results.items(), columns=["Var", "Aver"])
    return df.set_index("Var")["Aver"]


BASE_PATH = Path("C:/Users/rache/OneDrive - Cardiff University/MSc Bioinformatics/MET591 Dissertation/output")

sequences = [
    "APC_637_hot", "APC_641_non", "APC_3335_non", "APC_3340_hot",
    "APC_4099_hot", "APC_4103_non", "APC_4343_non", "APC_4348_hot",
    "TP53_632_non", "TP53_637_hot", "TP53_844_hot", "TP53_849_non",
]
replicates = [1, 2, 3]

writer = pd.ExcelWriter(BASE_PATH/"all_simulations.xlsx")

for seq in sequences:
    prefix = "_".join(seq.split("_")[:2])

    for rep in replicates:
        print(f"========= PROCESSING: {seq} | REPLICATE: {rep} =========")

        sim_name = f"{prefix}_{rep}"
        curves_dir = BASE_PATH / seq / str(rep) / "curves"

        try:
            lis_files = sorted(
                curves_dir.glob(f"{sim_name}-*-*.lis"),
                key=lambda p: int(p.stem.split("-")[-2])
            )

            tables = [lis_parser(f) for f in lis_files]
            labels = ["-".join(f.stem.split("-")[-2:]) for f in lis_files]

            combined = pd.concat(tables, axis=1, keys=labels)
            combined.to_excel(writer, sheet_name=sim_name)

        except Exception as e:
            print(f"!!! ERROR: {seq} Replicate {rep} failed: {e} !!!")
            with open(BASE_PATH / "failed_runs.log", "a") as log:
                log.write(f"{seq} | Replicate {rep} | Failed: {e}\n")

writer.close()
print(">>> ALL SIMULATIONS PROCESSED <<<")