import pandas as pd

cache = {}
cache_hits = 0
df_size = 0


def calculate_career_win_rate(df, player_id, match_nr, tourney_date):
    global cache
    global cache_hits
    global df_size

    if len(cache) % 5000 == 0 and len(cache) > 0:
        print(f"Cache hit rate: {cache_hits / (cache_hits + len(cache))} ({cache_hits}, {len(cache)})", )
        print(f"Processed {(len(cache) + cache_hits) / df_size * 100}% of the data")
        print("=====================================")

    if (player_id, tourney_date, match_nr) in cache:
        cache_hits += 1
        return cache[(player_id, tourney_date, match_nr)]

    filtered_df = df[
        (df["tourney_date"] < tourney_date) & (df["match_num"] < match_nr) & ((df["winner_id"] == player_id) | (
                df["loser_id"] == player_id))]
    total_games_played = len(filtered_df)

    if total_games_played == 0:
        result = cache[(player_id, tourney_date, match_nr)] = (0, 0, 0, 0, 0)
    else:
        total_games_won = filtered_df["winner_id"].value_counts().get(player_id, 0)

        win_rate = total_games_won / total_games_played
        loss_rate = 1 - win_rate
        result = (win_rate, loss_rate, total_games_played, total_games_won, total_games_played - total_games_won)

    cache[(player_id, tourney_date, match_nr)] = result
    return result


def main():
    global df_size
    df = pd.read_csv("../data/atp_matches_till_2022.csv")
    df_size = len(df)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    df = df.sort_values(by=["tourney_date", "match_num"])

    df["winner_career_win_rate"] = df.apply(
        lambda row: calculate_career_win_rate(df, row["winner_id"], row["match_num"], row["tourney_date"]), axis=1)
    print("Done with winners")
    df["loser_career_win_rate"] = df.apply(
        lambda row: calculate_career_win_rate(df, row["loser_id"], row["match_num"], row["tourney_date"]), axis=1)
    print("Done with losers")

    df[[
        "winner_win_rate",
        "winner_loss_rate",
        "winner_total_games_played",
        "winner_total_games_won",
        "winner_total_games_lost"
    ]] = pd.DataFrame([*df["winner_career_win_rate"]], df.index)

    df[[
        "loser_win_rate",
        "loser_loss_rate",
        "loser_total_games_played",
        "loser_total_games_won",
        "loser_total_games_lost"
    ]] = pd.DataFrame([*df["loser_career_win_rate"]], df.index)

    df = df.drop(columns=["winner_career_win_rate", "loser_career_win_rate"])

    df.to_csv("../data/atp_matches_till_2022_with_career.csv", index=False)


if __name__ == "__main__":
    main()
