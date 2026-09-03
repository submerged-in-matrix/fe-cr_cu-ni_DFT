import re
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "lattice_parameter.csv"

N_ATOMS = 16
TAG_RE = re.compile(r"fe(\d{2})cr(\d{2})")

FINAL_COORDS_RE = re.compile(
    r"Begin final coordinates.*?CELL_PARAMETERS \(angstrom\)\s*\n"
    r"\s*([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s*\n"
    r"\s*([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)\s*\n"
    r"\s*([\-\d\.]+)\s+([\-\d\.]+)\s+([\-\d\.]+)",
    re.DOTALL,
)


def parse_tag(tag: str):
    m = TAG_RE.match(tag)
    if not m:
        raise ValueError(f"Could not parse tag: {tag}")
    n_fe, n_cr = int(m.group(1)), int(m.group(2))
    return n_cr, n_cr / N_ATOMS


def extract_lattice_parameter(vcr_path: Path):
    text = vcr_path.read_text(errors="ignore")
    m = FINAL_COORDS_RE.search(text)
    if not m:
        return None

    vecs = [float(x) for x in m.groups()]
    a1 = np.array(vecs[0:3])
    a2 = np.array(vecs[3:6])
    a3 = np.array(vecs[6:9])

    lengths = [np.linalg.norm(a1), np.linalg.norm(a2), np.linalg.norm(a3)]
    max_dev = (max(lengths) - min(lengths)) / np.mean(lengths)
    is_cubic = max_dev < 0.01
    a_volume_equiv = (lengths[0] * lengths[1] * lengths[2]) ** (1.0 / 3.0)

    return {
        "a_angstrom": a_volume_equiv,
        "a_arithmetic_mean": np.mean(lengths),
        "a1": lengths[0], "a2": lengths[1], "a3": lengths[2],
        "cubic_deviation_pct": max_dev * 100,
        "is_cubic": is_cubic,
    }


def run():
    rows = []
    vcr_files = sorted(OUTPUTS_DIR.glob("fecr_*_vcr.out"))

    if not vcr_files:
        print(f"No fecr_*_vcr.out files found in {OUTPUTS_DIR} — check the path.")
        return

    for vcr_path in vcr_files:
        m = re.search(r"fecr_(fe\d{2}cr\d{2})_vcr\.out", vcr_path.name)
        if not m:
            continue
        tag = m.group(1)
        n_cr, x_cr = parse_tag(tag)

        result = extract_lattice_parameter(vcr_path)
        if result is None:
            print(f"WARNING: could not find final cell parameters in {vcr_path.name}")
            continue

        if not result["is_cubic"]:
            print(f"WARNING: {tag} final cell is non-cubic "
                  f"(deviation {result['cubic_deviation_pct']:.2f}%) — check this tag")

        rows.append({"tag": tag, "n_cr": n_cr, "x_cr": x_cr, **result})

    df = pd.DataFrame(rows).sort_values("x_cr").reset_index(drop=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {RESULTS_CSV}")
    print(df[["tag", "x_cr", "a_angstrom", "cubic_deviation_pct"]].to_string(index=False))


if __name__ == "__main__":
    run()