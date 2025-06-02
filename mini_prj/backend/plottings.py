import pandas as pd
import folium

# Load the dataset
df = pd.read_csv('cleaned_tracking_data.csv')

# 🔥 Ask user for bird ID
bird_id = input("Enter the Bird Identifier (leave empty to plot all birds): ").strip()

# Filter based on bird_id
if bird_id:
    bird_df = df[df['individual-local-identifier'] == bird_id]
else:
    bird_df = df.copy()  # Plot all birds if no ID is provided

# Check if bird_df is empty
if bird_df.empty:
    print("❌ No data found for the specified bird.")
    exit()

# Sort by timestamp
bird_df['timestamp'] = pd.to_datetime(bird_df['timestamp'])
bird_df = bird_df.sort_values('timestamp')

# Create a map centered at the first location
start_location = [bird_df.iloc[0]['location-lat'], bird_df.iloc[0]['location-long']]
m = folium.Map(location=start_location, zoom_start=6)

# Create list of (lat, lon) points
locations = list(zip(bird_df['location-lat'], bird_df['location-long']))

# Draw the path
folium.PolyLine(locations, color="blue", weight=2.5, opacity=1).add_to(m)

# Mark start and end points
folium.Marker(locations[0], tooltip="Start", icon=folium.Icon(color='green')).add_to(m)
folium.Marker(locations[-1], tooltip="End", icon=folium.Icon(color='red')).add_to(m)

# Save the map
m.save('bird_path_map.html')

print("✅ Map created! Open 'bird_path_map.html' in your browser.")
