import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import streamlit as st
import pydeck as pdk
from scipy.spatial import distance_matrix
from timeit import timeit

MILES_CONVERSION = 69 * 1.30

# TO DO:
# Add 2 more data visualizations. IDEAS:
#   Distance traveled by each truck
#   TIME when all deliveries are completed
#   Potentially allow the user to save the state of their settings, and use that as a data point?
#   I like that idea but it sounds like too much work :/
#   Potentially something related to packages? Like packages that each truck carried or something
#   -- Pretty sure I'll do this one


#                      Latitude    Longitude   Number of Packages
locations = np.array([[40.684770, -111.871110, 0],
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


algorithm = st.sidebar.selectbox("Select an Algorithm", ("Nearest Neighbor", "Nearest Neighbor 2-opt", "Nearest Insertion", "Farthest Insertion"))

if 'random_locations' not in st.session_state:
    st.session_state.random_locations = locations

# Layout setup
st.title("Truck Routing Optimization")

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
    temporary_distance = np.copy(distances)  # Create a deep copy of distances

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
        return None

    # Update route and visited set
    route.append(nearest_index)
    visited.add(nearest_index)

    # Reset distances to its original values
    distances[:] = temporary_distance  # Use slicing to update in-place

    return nearest_index


def two_opt(route):
    def calculate_route_distance(route):
        """Calculate the total distance of the given route based on the distance matrix."""
        distance = 0
        for i in range(len(route)):
            distance += dMatrix.iloc[route[i]][route[(i + 1) % len(route)]]
        return distance

    def reverse_segment(route, i, j):
        """Reverse the segment of the route from index i to index j."""
        new_route = route[:]
        new_route[i:j + 1] = reversed(new_route[i:j + 1])
        return new_route

    improved = True
    while improved:
        improved = False
        best_distance = calculate_route_distance(route)
        best_route = route[:]

        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                if j - i == 1:  # Skip adjacent pairs
                    continue

                new_route = reverse_segment(route, i, j)
                new_distance = calculate_route_distance(new_route)

                if new_distance < best_distance:
                    best_route = new_route[:]
                    best_distance = new_distance
                    improved = True

        route = best_route

    return route


def plot_nearest_insertion(current_index, visited, route):
    if len(route) == 1:
        # Start with the first insertion
        distances = dMatrix.iloc[current_index].values

        # Initialize variables to find the nearest city
        nearest_index = None
        nearest_distance = float('inf')

        # Iterate over all cities to find the nearest unvisited city
        for i in range(len(coordinates)):
            if i not in visited and distances[i] < nearest_distance:
                nearest_index = i
                nearest_distance = distances[i]

        if nearest_distance == float('inf'):
            # If all distances are inf, return None
            return None, float('inf')

        if nearest_distance is None:
            return None
        route.append(nearest_index)
        route.append(0)  # Append the hub
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


def plot_farthest_insertion(current_index, visited, route):
    if len(route) == 1:
        # Start with the first insertion
        distances = dMatrix.iloc[current_index].values

        # Initialize variables to find the nearest city
        farthest_index = None
        farthest_distance = 0

        # Iterate over all cities to find the furthest unvisited city
        for i in range(len(distances)):
            if i not in visited and distances[i] > farthest_distance:
                farthest_index = i
                farthest_distance = distances[i]

        if farthest_distance == 0:
            # If all distances are zero, return None
            return None, 0

        if farthest_distance is None:
            return None

        route.append(farthest_index)
        route.append(0) # Append the hub
        visited.add(farthest_index)
        return farthest_index
    else:
        distances = dMatrix.iloc[current_index].values

        best_insertion_index = None
        best_insertion_position = None
        best_insertion_distance = 0

        for candidate in range(len(coordinates)):
            if candidate not in visited:
                # Calculate the best insertion position
                if candidate not in visited and distances[candidate] > best_insertion_distance:
                    best_insertion_index = candidate
                best_distance = float('inf')
                #Calculate the best insertion point
                for i in range(len(route)):
                    start = route[i]
                    end = route[(i + 1) % len(route)]
                    new_distance = dMatrix.iloc[start][candidate] + dMatrix.iloc[candidate][end] - \
                                   dMatrix.iloc[start][end]
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_insertion_position = i

        if best_insertion_index is None:
            return None

        # Insert the city into the best position
        route.insert(best_insertion_position + 1, best_insertion_index)
        visited.add(best_insertion_index)

        return best_insertion_index


truck_df = pd.DataFrame(columns=["Packages Delivered", "Distance Traversed (Miles)", "Time Elapsed (Hours)", "Wages Paid",
                                 "MPG", "Gallons of Gas Expended", "Average Speed"])


def calculate_route(number_of_iterations, number_of_trucks, capacity, selected_algorithm, colors):
    lines = pd.DataFrame(columns=["start_lat", "start_lon", "end_lat", "end_lon", "color", "length"])
    visited = set()
    visited.add(0)
    i = 0
    furthest_insertion = 0

    def append_route_to_lines(route, truck):
        nonlocal lines
        route_distance = 0

        for k in range(len(route) - 1):
            start_index = route[k]
            end_index = route[k + 1]
            distance = dMatrix.iloc[start_index][end_index]
            route_distance += distance
            lines.loc[len(lines)] = {
                'start_lat': coordinates[start_index][0],
                'start_lon': coordinates[start_index][1],
                'end_lat': coordinates[end_index][0],
                'end_lon': coordinates[end_index][1],
                'color': truck_colors.get(truck),
                'length': distance
            }

        # I still don't know why I have to divide the end result by 2, but it doesn't work without this

        return route_distance / 2

    def execute_algorithm(current_index, route, truck):
        nonlocal visited
        nonlocal selected_algorithm

        if selected_algorithm == 'Nearest Neighbor':
            return plot_nearest_neighbor(current_index, visited, route)
        elif selected_algorithm == 'Nearest Neighbor 2-opt':
            current_index = plot_nearest_neighbor(current_index, visited, route)
            route = two_opt(route)
            return route, current_index
        elif selected_algorithm == 'Nearest Insertion':
            return plot_nearest_insertion(current_index, visited, route)
        elif selected_algorithm == 'Farthest Insertion':
            return plot_farthest_insertion(current_index, visited, route)
        else:
            st.write("INVALID SELECTION")
            return None

    while True:
        terminate_early = False
        pos = 0

        for truck in range(number_of_trucks):
            packages = 0
            current_index = 0
            j = i
            route = [current_index]

            while len(visited) < len(coordinates) and packages < capacity and j < number_of_iterations:
                j += 1
                result = execute_algorithm(current_index, route, truck)
                if isinstance(result, tuple):  # For algorithms that return both route and index
                    route, current_index = result
                else:
                    current_index = result
                packages += random_locations[current_index, 2]
                if j >= number_of_iterations or packages >= capacity or len(visited) >= len(coordinates):
                    route.append(0)
                    break
                if current_index is None:
                    break

            route_distance = append_route_to_lines(route, truck)

            truck_index = f'truck {truck + 1}'

            global truck_df

            if truck_index not in truck_df.index:
                # Create the DataFrame for the new truck
                temp_truck_df = pd.DataFrame({
                    "Packages Delivered": packages / 2,
                    "Distance Traversed (Miles)": route_distance * 69 * 1.30,
                    "Time Elapsed (Hours)": "",
                    "Wages Paid": "",
                    "MPG": 10,
                    "Gallons of Gas Expended": None,
                    "Average Speed": 13.6},
                    index=[truck_index])

                # Perform calculations
                temp_truck_df["Time Elapsed (Hours)"] = temp_truck_df["Distance Traversed (Miles)"] / temp_truck_df["Average Speed"]
                temp_truck_df["Wages Paid"] = temp_truck_df["Time Elapsed (Hours)"] * 18.76
                temp_truck_df["Gallons of Gas Expended"] = temp_truck_df["Distance Traversed (Miles)"] / temp_truck_df["MPG"]

                # Concatenate the new data with the existing DataFrame
                truck_df = pd.concat([truck_df, temp_truck_df])
            else:
                truck_df.loc[truck_index, "Packages Delivered"] += packages / 2
                truck_df.loc[truck_index, "Distance Traversed (Miles)"] += route_distance * 69 * 1.30
                truck_df.loc[truck_index, "Time Elapsed (Hours)"] = truck_df.loc[truck_index, "Distance Traversed (Miles)"] / \
                                                            truck_df.loc[truck_index, "Average Speed"]
                truck_df.loc[truck_index, "Wages Paid"] = truck_df.loc[truck_index, "Time Elapsed (Hours)"] * 18.76
                truck_df.loc[truck_index, "Gallons of Gas Expended"] = truck_df.loc[truck_index, "Distance Traversed (Miles)"] / \
                                                                       truck_df.loc[truck_index, "MPG"]

            if j == number_of_iterations:
                terminate_early = True

            pos += 1

            if furthest_insertion < j:
                furthest_insertion = j

            if pos == number_of_trucks:
                i = furthest_insertion

        if len(visited) == len(coordinates) or terminate_early:
            break

    return lines


st.sidebar.header("Settings")
n_trucks = st.sidebar.slider("Number of Trucks", min_value=1, max_value=5, value=1)
truck_capacity = st.sidebar.slider("Number of Packages per Truck", min_value=5, max_value=100, value=20)
n_iterations = st.sidebar.slider("Nearest Neighbor Iterations", min_value=0, max_value=-(-len(locations) // n_trucks), value=0)
random_dataset_size = st.sidebar.slider("Size of the Randomized Dataset", min_value=10, max_value=100, value=20)

if st.sidebar.button("Randomize Dataset"):
    st.session_state.random_locations = generate_random_locations(random_dataset_size, locations)
    random_locations = st.session_state.random_locations
    df = pd.DataFrame(
        {
            "lat": random_locations[:, 0],
            "lon": random_locations[:, 1],
            "num_packages": random_locations[:, 2],
            "radius": 100 + (random_locations[:, 2] * 25),
            "col4": "#f00",
        }
    )

truck_colors = {
    0: (31, 119, 180, 255),  # Truck 1 - Blue
    1: (255, 127, 14, 255),  # Truck 2 - Orange
    2: (44, 160, 44, 255),   # Truck 3 - Green
    3: (214, 39, 40, 255),   # Truck 4 - Red
    4: (148, 103, 189, 255), # Truck 5 - Purple
}

truck_colors_html = {
    0: '#1f77b4',  # Truck 1 - Blue
    1: '#ff7f0e',  # Truck 2 - Orange
    2: '#2ca02c',  # Truck 3 - Green
    3: '#d62728',  # Truck 4 - Red
    4: '#9467bd',  # Truck 5 - Purple
}

route_df = calculate_route(n_iterations, n_trucks, truck_capacity, algorithm, truck_colors)
total_distance = route_df['length'].sum() * MILES_CONVERSION


def wrapper():
    calculate_route(n_iterations, n_trucks, truck_capacity, algorithm, truck_colors)


# Measure the time
elapsed_time = timeit(wrapper, number=1)

# Display the time in Streamlit
st.header("Route Analysis")
col1, col2 = st.columns([3, 1])

with col1:
    st.pydeck_chart(pdk.Deck(
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
    ))

with col2:
    st.write(f"**Total Distance Traveled:** {total_distance:.2f} miles")
    for i in range(n_trucks):
        st.markdown(
            f"""
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="width: 20px; height: 20px; background-color: {truck_colors_html[i]}; border: 1px solid #000; margin-right: 10px;"></div>
                    <div>Truck {i + 1}</div>
                </div>
            """,
            unsafe_allow_html=True
        )


st.header("Truck Data")
options = st.multiselect(
    "What would you like to Display?",
    ["Packages Delivered", "Distance Traversed (Miles)", "Time Elapsed (Hours)", "Wages Paid", "Gallons of Gas Expended"],
    ["Packages Delivered", "Distance Traversed (Miles)"],
)

st.bar_chart(truck_df[options], horizontal=True, stack=False)

columns_of_interest = ["Packages Delivered", "Distance Traversed (Miles)", "Time Elapsed (Hours)", "Wages Paid", "Gallons of Gas Expended"]
subset_df = truck_df[columns_of_interest]

min_values = subset_df.min()
max_values = subset_df.max()
average_values = subset_df.mean()
total_values = subset_df.sum()


st.header("Summary Statistics")
requested_query = st.selectbox("What would you like to Query?", ("Averages", "Minimums", "Maximums", "Totals"))

if requested_query == "Averages":
    st.table(average_values)
elif requested_query == "Minimums":
    st.table(min_values)
elif requested_query == "Maximums":
    st.table(max_values)
elif requested_query == "Totals":
    st.table(total_values)

# Measure the time
elapsed_time = timeit(lambda: calculate_route(n_iterations, n_trucks, truck_capacity, algorithm, truck_colors), number=1)
st.write(f"**Elapsed time:** {elapsed_time:.6f} seconds")