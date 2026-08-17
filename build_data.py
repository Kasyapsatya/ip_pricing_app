"""
Run this ONCE (locally, before deploying) to generate the pricing artifacts file
that the Streamlit app loads at startup:

    python build_data.py

This simulates the population + spells and computes all three tool tables, then
pickles the result to data/pricing_artifacts.pkl. The app never regenerates this
data itself — it just loads the file. Re-run this script only if you deliberately
want a fresh dataset (e.g. after changing a constant in pricing_tools.py).
"""
import pickle
from pathlib import Path

import pricing_tools as pt

OUTPUT_PATH = Path(__file__).parent / "data" / "pricing_artifacts.pkl"


def main():
    print("Generating population and simulating sickness spells...")
    artifacts = pt.build_artifacts()

    print(f"  population: {len(artifacts['population']):,} lives")
    print(f"  spells:     {len(artifacts['spells']):,} sickness spells")
    print(f"  base_table (Tool 1): {len(artifacts['base_table']['incidence_table'])} age x occupation cells")
    print(f"  deferred_period_table (Tool 2): {len(artifacts['deferred_period_table'])} deferred-period options")
    print(f"  loading_table (Tool 3): {len(artifacts['loading_table'])} episode bands")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
