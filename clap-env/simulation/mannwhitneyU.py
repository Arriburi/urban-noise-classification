import pandas as pd
from scipy.stats import mannwhitneyu


RANDOM_PATH = "/home/arriburi/projects/urban-noise-classification/clap-env/simulation/distribution_results/random.csv"
GT15_PATH = "/home/arriburi/projects/urban-noise-classification/clap-env/simulation/distribution_results/groundtruth_15.csv"


def load_entropy_values(csv_path: str) -> list[float]:
    df = pd.read_csv(csv_path, index_col=0)
    entropy_row = df.loc["ENTROPY"]
    values: list[float] = []
    for col in entropy_row.index:
        if col.startswith("seed"):
            values.append(float(entropy_row[col]))
    return values


def main():

    groupA = load_entropy_values(GT15_PATH)
    groupB = load_entropy_values(RANDOM_PATH)

    u_stat, p_value = mannwhitneyu(groupA, groupB, alternative="greater")
    print("\nMann-Whitney U test (entropy, GT > random):")
    print("U statistic:", u_stat)
    print("p-value:", p_value)


if __name__ == "__main__":
    main()
