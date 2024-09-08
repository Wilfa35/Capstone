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
    # NEAREST INSERTION -- DONE
    # FURTHEST INSERTION -- TBC
    # 2-OPT -- TBC
    # GREEDY INSERTION -- TBC

# ALLOW THE USER to RANDOMIZE the DATASET -- DONE
# ALLOW THE USER to change the NUMBER OF TRUCKS and PACKAGES -- DONE
# (POTENTIALLY) USE AN ANIMATION as opposed to THE SLIDER
# WRITE OR INCORPORATE IT THIS INTO ROUTING FUNCTION -- WE NEED TO MEASURE THE DISTANCE OF THE ROUTE (IMPORTANT)

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


def generate_random_locations(num_rows, original_array):
    # Extract the minimum and maximum latitude and longitude from the original array
    min_latitude, max_latitude = original_array[:, 0].min(), original_array[:, 0].max()
    min_longitude, max_longitude = original_array[:, 1].min(), original_array[:, 1].max()

    # Determine the range of the number of packages
    min_packages, max_packages = original_array[:, 2].min(), original_array[:, 2].max()

    # Generate random latitudes and longitudes within the bounds
    random_latitudes = np.random.uniform(min_latitude, max_latitude, num_rows)
    random_longitudes = np.random.uniform(min_longitude, max_longitude, num_rows)

    # Generate random number of packages within the bounds
    random_packages = np.random.randint(min_packages, max_packages + 1, num_rows)

    # Combine into a single array
    random_locations = np.column_stack((random_latitudes, random_longitudes, random_packages))

    print(random_locations)
    return random_locations


algorithm = st.selectbox("Select an Algorithm", ("Nearest Neighbor", "Nearest Neighbor 2-opt", "Nearest Insertion"))

if 'random_locations' not in st.session_state:
    st.session_state.random_locations = locations

# Button to randomize dataset
if st.button("Randomize Dataset"):
    st.session_state.random_locations = generate_random_locations(20, locations)

# Access the randomized locations from session state
random_locations = st.session_state.random_locations

# Create a DataFrame for display
df = pd.DataFrame(
    {
        "lat": random_locations[:, 0],
        "lon": random_locations[:, 1],
        "num_packages": random_locations[:, 2],
        "radius": 100 + (random_locations[:, 2] * 25),
        "col4": "#f00",
    }
)

coordinates = random_locations[:, :2]

dMatrix = pd.DataFrame(distance_matrix(coordinates, coordinates))


def plot_nearest_neighbor(current_index, visited, route):
    # Get the distances for the current index
    distances = dMatrix.iloc[current_index].values
    # Set the distance to self to infinity
    distances[current_index] = float('inf')
    # Exclude already visited nodes by setting their distances to infinity
    for v in visited:
        distances[v] = float('inf')
    # Get the index of the nearest neighbor
    nearest_index = np.argmin(distances)
    nearest_distance = distances[nearest_index]
    if nearest_distance == float('inf'):
        # If all distances are inf, return None
        return None, float('inf')
    if nearest_index in visited:
        # Find the next nearest neighbor that has not been visited
        while nearest_index in visited:
            next_index, distance = plot_nearest_neighbor(nearest_index, visited, route)
    route.append(nearest_index)
    visited.add(nearest_index)
    return nearest_index


def plot_nearest_insertion(current_index, visited, route):
    if len(route) == 1:
        # Start with the first insertion
        distances = dMatrix.iloc[current_index].values

        # Initialize variables to find the nearest city
        nearest_index = None
        nearest_distance = float('inf')

        # Iterate over all cities to find the nearest unvisited city
        for i in range(len(distances)):
            if i not in visited and distances[i] < nearest_distance:
                nearest_index = i
                nearest_distance = distances[i]

        if nearest_distance == float('inf'):
            # If all distances are inf, return None
            return None, float('inf')

        if nearest_distance is None:
            return None
        route.append(nearest_index)
        visited.add(nearest_index)
        return nearest_index
    else:
        best_insertion_index = None
        best_insertion_position = None
        best_insertion_distance = float('inf')

        for candidate in range(len(coordinates)):
            if candidate not in visited:
                # Calculate the best insertion point
                for i in range(len(route)):
                    start = route[i]
                    end = route[(i + 1) % len(route)]
                    new_distance = dMatrix.iloc[start][candidate] + dMatrix.iloc[candidate][end] - \
                                   dMatrix.iloc[start][end]
                    if new_distance < best_insertion_distance:
                        best_insertion_index = candidate
                        best_insertion_position = i
                        best_insertion_distance = new_distance

        if best_insertion_index is None:
            return None

        # Insert the city into the best position
        route.insert(best_insertion_position + 1, best_insertion_index)
        visited.add(best_insertion_index)
        return best_insertion_index
# Add an "algorithm" field to calculate route that decides which algorithm to run based on user selection
# Probably a switch case statement for running the algorithms based on user selection


def farthest_insertion(index, visited):
    # Get the distances for the current index
    distances = dMatrix.iloc[index].values

    # Initialize variables to find the nearest city
    nearest_index = None
    nearest_distance = float('inf')

    # Iterate over all cities to find the nearest unvisited city
    for i in range(len(distances)):
        if i not in visited and distances[i] < nearest_distance:
            nearest_index = i
            nearest_distance = distances[i]

    if nearest_distance == float('inf'):
        # If all distances are inf, return None
        return None, float('inf')

    return nearest_index, nearest_distance


def plot_farthest_insertion(current_index, visited, route):
    # Get the distances for the current index
    distances = dMatrix.iloc[current_index].values

    # Initialize variables to find the nearest city
    nearest_index = None
    nearest_distance = float('inf')

    # Iterate over all cities to find the nearest unvisited city
    for i in range(len(distances)):
        if i not in visited and distances[i] < nearest_distance:
            nearest_index = i
            nearest_distance = distances[i]

    if nearest_distance == float('inf'):
        # If all distances are inf, return None
        return None, float('inf')

    return nearest_index, nearest_distance


def calculate_route(number_of_iterations, number_of_trucks, capacity, selected_algorithm):
    lines = pd.DataFrame(columns=["start_lat", "start_lon", "end_lat", "end_lon", "color", "length"])
    visited = set()

    for truck in range(number_of_trucks):
        packages = 0
        current_index = 0  # Starting from the first coordinate -- the "hub"
        j = 0
        # Initial route setup
        route = [current_index]
        while len(visited) < len(coordinates) and packages < capacity and j < number_of_iterations:
            j += 1
            packages += locations[current_index, 2]
            if selected_algorithm == 'Nearest Neighbor':
                current_index = plot_nearest_neighbor(current_index, visited, route)
                if current_index is None:
                    break
            elif selected_algorithm == 'Nearest Insertion':
                current_index = plot_nearest_insertion(current_index, visited, route)
                if current_index is None:
                    break
            else:
                st.write("INVALID SELECTION")
                break
            # Append the route
        for i in range(len(route) - 1):
            start_index = route[i]
            end_index = route[i + 1]
            distance = dMatrix.iloc[start_index][end_index]
            lines.loc[len(lines)] = {
                'start_lat': coordinates[start_index][0],
                'start_lon': coordinates[start_index][1],
                'end_lat': coordinates[end_index][0],
                'end_lon': coordinates[end_index][1],
                'color': [250 - (truck * 50), 50 * truck, 25 * truck, 160],
                'length': distance
            }

    return lines


n_iterations = st.slider("Nearest Neighbor Iterations", min_value=0, max_value=len(coordinates) - 1, value=0,
                         help="How many iterations of your current algorithm")
n_trucks = st.slider("Number of Trucks", min_value=1, max_value=5, value=1, help="How many trucks")
truck_capacity = st.slider("Number of Packages per Truck", min_value=5, max_value=100, value=20, help="How many trucks")

route_df = calculate_route(n_iterations, n_trucks, truck_capacity, algorithm)

# ADD constants. The 69 is latitude to miles. 1.30 is the difference between the shortest route to a straight line.
# Include this source in blurb: https://blog.cdxtech.com/post/straight-line-distance-as-an-estimate-for-driving-routes
st.write(route_df['length'].sum() * 69 * 1.30)

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
