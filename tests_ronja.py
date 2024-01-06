import pandas as pd
from geopy.geocoders import MapBox

# Read the CSV file
df = pd.read_csv('/home/ronja/documents/studium/bi-project/data/atp_tourneys_till_2022.csv', sep=';')

# Create a geocoder object
geolocator = MapBox(api_key='pk.eyJ1Ijoicm9uamFzIiwiYSI6ImNscjI1bjZvMzB3dDAya3A4d2dieXkzbDMifQ.UYrZLJc6JNKgJdDzcqifNA')

print(df.head())

# Create a dictionary to store the coordinates for each location
location_coordinates = {}

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    location = row['tourney_location']

    # Check if the location contains 'davis'
    if 'Davis' in location:
        continue

    # Check if the location is already in the dictionary
    if location in location_coordinates:
        latitude, longitude = location_coordinates[location]
    else:
        # Geocode the location to get the coordinates
        try:
            coordinates = geolocator.geocode(location)
            latitude = coordinates.latitude
            longitude = coordinates.longitude

            # Store the coordinates in the dictionary
            location_coordinates[location] = (latitude, longitude)
        except Exception as e:
            print(f"Error occurred while geocoding location {location}: {e}")
            continue

    # Update the corresponding row in the DataFrame
    df.at[index, 'latitude'] = latitude
    df.at[index, 'longitude'] = longitude

# Save the updated DataFrame back to the CSV file
df.to_csv('/home/ronja/documents/studium/bi-project/data/atp_tourneys_till_2022.csv', index=False)
