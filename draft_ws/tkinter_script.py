import random

# Approximate boundaries of Riverside, California
# Latitude range: 33.70 to 34.10
# Longitude range: -117.45 to -117.20

def generate_riverside_coordinates():
    """Generate a random coordinate within Riverside, California."""
    latitude = round(random.uniform(33.70, 34.10), 5)
    longitude = round(random.uniform(-117.45, -117.20), 5)
    return latitude, longitude

# Generate and print a single coordinate
lat, lon = generate_riverside_coordinates()
print(f"Coordinate: {lat}, {lon}")