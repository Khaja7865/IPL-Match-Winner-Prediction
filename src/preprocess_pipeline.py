import os
import pandas as pd

DATA_PATH = os.path.join(
    "Dataset", "Raw data", "matches (1).csv"
)

OUTPUT_DIR = "outputs"


def create_historical_features(matches):
    matches = matches.dropna(subset=["winner"]).copy()

    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    team_wins = {}
    team_games = {}

    team1_rates = []
    team2_rates = []

    for _, row in matches.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]

        team1_rates.append(
            team_wins.get(team1, 0) /
            max(team_games.get(team1, 0), 1)
        )

        team2_rates.append(
            team_wins.get(team2, 0) /
            max(team_games.get(team2, 0), 1)
        )

        team_games[team1] = team_games.get(team1, 0) + 1
        team_games[team2] = team_games.get(team2, 0) + 1

        if row["winner"] == team1:
            team_wins[team1] = team_wins.get(team1, 0) + 1
        elif row["winner"] == team2:
            team_wins[team2] = team_wins.get(team2, 0) + 1

    matches["team1_win_rate"] = team1_rates
    matches["team2_win_rate"] = team2_rates

    matches["team1_won"] = (
        matches["winner"] == matches["team1"]
    ).astype(int)

    return matches


def main():
    print("[INFO] Running Lab 5 preprocessing...")

    matches = pd.read_csv(DATA_PATH)
    matches = create_historical_features(matches)

    features = [
        "season",
        "city",
        "venue",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "team1_win_rate",
        "team2_win_rate"
    ]

    X = matches[features].copy()
    y = matches["team1_won"].copy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X.to_csv(
        os.path.join(OUTPUT_DIR, "X_pipeline.csv"),
        index=False
    )

    y.to_csv(
        os.path.join(OUTPUT_DIR, "y_pipeline.csv"),
        index=False
    )

    print("[INFO] Pipeline X shape:", X.shape)
    print("[INFO] Pipeline y shape:", y.shape)

    print("[SUCCESS] Lab 5 preprocessing completed.")


if __name__ == "__main__":
    main()