import pandas as pd


def analyze_salt_matches(input_path: str):
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    
    print(f"\nInput file: {input_path}")
    print(f"Total rows: {total_rows}")
    
    invalid_mask = df["salt_match"].isna()
    invalid_count = invalid_mask.sum()
    df_valid = df[~invalid_mask]
    valid_count = len(df_valid)
    
    print(f"\n--- Validity Breakdown ---")
    print(f"Valid labels:   {valid_count:>6} ({100 * valid_count / total_rows:.2f}%)")
    print(f"Invalid labels: {invalid_count:>6} ({100 * invalid_count / total_rows:.2f}%)")
    
    true_count = df_valid["salt_match"].sum()
    false_count = valid_count - true_count
    
    accuracy = true_count / valid_count if valid_count > 0 else 0.0
    
    print(f"\n--- Match Results (Valid Only) ---")
    print(f"Matched (True):      {true_count:>6} ({100 * true_count / valid_count:.2f}%)")
    print(f"Not Matched (False): {false_count:>6} ({100 * false_count / valid_count:.2f}%)")
    print(f"\nCLAP Accuracy: {100 * accuracy:.2f}%")
    
    return {
        "total_rows": total_rows,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "matched_count": true_count,
        "not_matched_count": false_count,
        "accuracy": accuracy,
    }


def main():
    input_path = "/home/lucaa/urban-noise-classification/clap-env/simulation/salt_matches.parquet"
    stats = analyze_salt_matches(input_path)
    return stats


if __name__ == "__main__":
    main()

