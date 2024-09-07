import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import streamlit as st
import pydeck as pdk
from scipy.spatial import distance_matrix

# TO DO:
# Add the PACKAGES and TRUCKS feature -- DONE
# WRITE A BLURB explaining the dataset, scenario, and how the dataset can be visualized
# ADD INTERACTIVE ELEMENTS such that the user can see the difference between the 2-opt, greedy, nearest insertion, ect.
# https://stemlounge.com/animated-algorithms-for-the-traveling-salesman-problem/
# ALLOW THE USER to RANDOMIZE the DATASET
# ALLOW THE USER to change the NUMBER OF TRUCKS and PACKAGES
# (POTENTIALLY) USE AN ANIMATION as opposed to THE SLIDER

#                      Latitude    Longitude   Number of Packages
locations = np.array([[40.684770, -111.871110, 1],
                      [40.745930, -111.938160, 2],
                      [40.725330, -111.852966, 3],
                      [40.664470, -111.933160, 1],
                      [40.692458, -111.895690, 2],
                      [40.716805, -111.895319, 4],
                      [40.758513, -111.947929, 2],
                      [40.712312, -111.954971, 1],
                      [40.774841, -111.885948, 2],
                      [40.715443, -111.877740, 1],
                      [40.654498, -111.957709, 3],
                      [40.710026, -111.890679, 4],
                      [40.775667, -111.888137, 1],
                      [40.704667, -111.937086, 1],
                      [40.702483, -111.922838, 5],
                      [40.697871, -111.916820, 1],
                      [40.703364, -111.909673, 1],
                      [40.691036, -111.891020, 3],
                      [40.708705, -111.901988, 2],
                      [40.760331, -111.888332, 3],
                      [40.676338, -111.854214, 1],
                      [40.671442, -111.824646, 5],
                      [40.662317, -111.887077, 5],
                      [40.659050, -111.958002, 2],
                      [40.653966, -111.865307, 1],
                      [40.749703, -111.873752, 2],
                      [40.635141, -111.864093, 2]])

df = pd.DataFrame(
    {
        "lat": locations[:, 0],
        "lon": locations[:, 1],
        "num_packages": locations[:, 2],
        "radius": 100 + (locations[:, 2] * 25),
        "col4": "#f00",
    }
)

coordinates = locations[:, :2]

dMatrix = pd.DataFrame(distance_matrix(coordinates, coordinates))


def nearest_neighbor(index, visited):
    # Get the distances for the current index
    distances = dMatrix.iloc[index].values
    # Set the distance to self to infinity
    distances[index] = float('inf')
    # Exclude already visited nodes by setting their distances to infinity
    for v in visited:
        distances[v] = float('inf')
    # Get the index of the nearest neighbor
    nearest_index = np.argmin(distances)
    nearest_distance = distances[nearest_index]
    if nearest_distance == float('inf'):
        # If all distances are inf, return None
        return None, float('inf')
    return nearest_index, nearest_distance


def calculate_route(number_of_iterations, number_of_trucks, capacity):
    lines = pd.DataFrame(columns=["start_lat", "start_lon", "end_lat", "end_lon", "color"])
    visited = set()

    for truck in range(number_of_trucks):
        packages = 0
        current_index = 0  # Starting from the first coordinate
        i = 0
        while len(visited) < len(coordinates) and i < number_of_iterations and packages < capacity:
            if current_index in visited and current_index is not 0:
                break
            i += 1
            visited.add(current_index)
            packages += locations[current_index, 2]
            # Find the nearest neighbor for the current coordinate
            next_index, distance = nearest_neighbor(current_index, visited)
            if next_index is None:
                break
            if next_index in visited:
                # Find the next nearest neighbor that has not been visited
                while next_index in visited:
                    next_index, distance = nearest_neighbor(next_index, visited)
            # Append the route
            lines.loc[len(lines)] = {
                'start_lat': coordinates[current_index][0],
                'start_lon': coordinates[current_index][1],
                'end_lat': coordinates[next_index][0],
                'end_lon': coordinates[next_index][1],
                'color': [250 - (truck * 50), 50 * truck, 100, 160]
            }
            current_index = next_index  # Move to the next coordinate

    return lines


print(dMatrix.to_string())

n_iterations = st.slider("Nearest Neighbor Iterations", min_value=0, max_value=len(coordinates) - 1, value=0,
                         help="How many iterations of your current algorithm")
n_trucks = st.slider("Number of Trucks", min_value=1, max_value=5, value=1, help="How many trucks")
truck_capacity = st.slider("Number of Packages per Truck", min_value=5, max_value=100, value=20, help="How many trucks")

route_df = calculate_route(n_iterations, n_trucks, truck_capacity)

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=40.705,
            longitude=-111.90,
            zoom=10.75,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                get_radius="radius",
            ),
            pdk.Layer(
                "LineLayer",
                route_df,
                get_source_position="[start_lon, start_lat]",
                get_target_position="[end_lon, end_lat]",
                get_color="color",
                get_width=5,
                highlight_color=[255, 255, 0],
                auto_highlight=True,
                pickable=False,
            ),
        ],
    )
)
