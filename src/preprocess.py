import os
import pandas as pd


DATA_PATH = os.path.join(
    "Dataset",
    "Raw data",
    "matches (1).csv"
)

OUTPUT_DIR = "outputs"

FEATURES = [
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


def create_historical_features(matches):
    matches = matches.dropna(subset=["winner"]).copy()

    matches["date"] = pd.to_datetime(matches["date"])

    # Sort chronologically so only previous matches
    # are used to calculate historical performance.
    matches = matches.sort_values("date").reset_index(drop=True)

    team_wins = {}
    team_games = {}

    team1_rates = []
    team2_rates = []

    for _, row in matches.iterrows():

        team1 = row["team1"]
        team2 = row["team2"]

        team1_rates.append(
            team_wins.get(team1, 0)
            / max(team_games.get(team1, 0), 1)
        )

        team2_rates.append(
            team_wins.get(team2, 0)
            / max(team_games.get(team2, 0), 1)
        )

        # Update history after the match
        team_games[team1] = team_games.get(team1, 0) + 1
        team_games[team2] = team_games.get(team2, 0) + 1

        if row["winner"] == team1:
            team_wins[team1] = team_wins.get(team1, 0) + 1

        elif row["winner"] == team2:
            team_wins[team2] = team_wins.get(team2, 0) + 1

    matches["team1_win_rate"] = team1_rates
    matches["team2_win_rate"] = team2_rates

    # Target: 1 if Team 1 won, otherwise 0
    matches["team1_won"] = (
        matches["winner"] == matches["team1"]
    ).astype(int)

    return matches


def main():

    print("[INFO] Loading IPL dataset...")

    matches = pd.read_csv(DATA_PATH)

    print(f"[INFO] Original shape: {matches.shape}")

    matches = create_historical_features(matches)

    print(f"[INFO] Usable matches: {len(matches)}")

    X = matches[FEATURES].copy()
    y = matches["team1_won"].copy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X.to_csv(
        os.path.join(OUTPUT_DIR, "X_ipl.csv"),
        index=False
    )

    y.to_csv(
        os.path.join(OUTPUT_DIR, "y_ipl.csv"),
        index=False
    )

    processed = matches[FEATURES + ["team1_won"]]

    processed.to_csv(
        os.path.join(OUTPUT_DIR, "processed_matches.csv"),
        index=False
    )

    print("[INFO] Features:")
    print(FEATURES)

    print("[INFO] X shape:", X.shape)
    print("[INFO] y shape:", y.shape)

    print("[SUCCESS] IPL preprocessing completed.")


if __name__ == "__main__":
    main()