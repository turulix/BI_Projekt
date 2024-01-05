import pandas as pd

# Read the CSV file with the correct delimiter
df = pd.read_csv('/home/ronja/documents/studium/bi-project/data/atp_matches_till_2022_preprocessed.csv', delimiter=';', low_memory=False)

# Convert 'tourney_date' column to datetime
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')

# Sort by 'tourney_date'
df = df.sort_values('tourney_date')

# Extract the year from 'tourney_date' and create a new column 'year'
df['year'] = df['tourney_date'].dt.year


# Create a DataFrame for matches played by each player
df_player = pd.concat([df[['year', 'tourney_date', 'winner_id']].rename(columns={'winner_id': 'player_id'}), 
                       df[['year', 'tourney_date', 'loser_id']].rename(columns={'loser_id': 'player_id'})])

# Sort by 'year', 'player_id' and 'tourney_date'
df_player = df_player.sort_values(['year', 'player_id', 'tourney_date'])

# Calculate the cumulative count of matches for each player each year
df_player['cumulative_matches'] = df_player.groupby(['year', 'player_id']).cumcount() + 1

# Calculate the total number of matches played by each player each year
df_player['total_matches'] = df_player.groupby(['year', 'player_id'])['player_id'].transform('count')

# Calculate the ratio of cumulative matches to total matches
df_player['match_ratio'] = df_player['cumulative_matches'] / df_player['total_matches']

# Merge the match_ratio for the winner and the loser back into the original DataFrame
df = pd.merge(df, df_player[['year', 'tourney_date', 'player_id', 'match_ratio']], 
              how='left', left_on=['year', 'tourney_date', 'winner_id'], right_on=['year', 'tourney_date', 'player_id'])
df = df.rename(columns={'match_ratio': 'winner_match_ratio'}).drop(columns='player_id')

df = pd.merge(df, df_player[['year', 'tourney_date', 'player_id', 'match_ratio']], 
              how='left', left_on=['year', 'tourney_date', 'loser_id'], right_on=['year', 'tourney_date', 'player_id'])
df = df.rename(columns={'match_ratio': 'loser_match_ratio'}).drop(columns='player_id')


# Save the modified DataFrame back to the CSV file
df.to_csv('/home/ronja/documents/studium/bi-project/data/atp_matches_till_2022_preprocessed_2.csv', index=False)
