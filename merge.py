import os
import pandas as pd
import pm4py

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATA_DIR = "Project/COVAS"

INPUT_FILES = [
    os.path.join(DATA_DIR, "FIRST_WAVE.xes"),
    os.path.join(DATA_DIR, "SECOND_WAVE.xes"),
    os.path.join(DATA_DIR, "THIRD_WAVE.xes"),
]

OUTPUT_FILE = os.path.join("Data", "all_waves_merged.xes")


def merge_xes_logs(input_files, output_file):
    """
    Loads multiple XES event logs and merges them into a single XES file.
    Cases/events are simply concatenated, no extra wave marker is added.
    """
    all_dfs = []

    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find file: {file_path}")

        print(f"Loading '{file_path}'...")
        log = pm4py.read_xes(file_path)
        df = pm4py.convert_to_dataframe(log)
        print(f" -> Loaded {df['case:concept:name'].nunique()} cases, {len(df)} events.")
        all_dfs.append(df)

    print("\nConcatenating logs...")
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Keep traces well-ordered: group by case, order events by timestamp
    if "case:concept:name" in merged_df.columns and "time:timestamp" in merged_df.columns:
        merged_df = merged_df.sort_values(
            by=["case:concept:name", "time:timestamp"]
        ).reset_index(drop=True)

    print(f"Total merged: {merged_df['case:concept:name'].nunique()} cases, {len(merged_df)} events.")

    print(f"\nWriting merged log to '{output_file}'...")
    pm4py.write_xes(merged_df, output_file)
    print("Done!")

    return merged_df


if __name__ == "__main__":
    print(f"Merging {len(INPUT_FILES)} XES files...\n")
    merge_xes_logs(INPUT_FILES, OUTPUT_FILE)