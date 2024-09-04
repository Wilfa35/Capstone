import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pydeck as pdk

coords = np.array([[40.684770, -111.871110],
                   [40.745930, -111.938160],
                   [40.725330, -111.852966],
                   [40.664470, -111.933160],
                   [40.692458, -111.895690],
                   [40.716805, -111.895319],
                   [40.758513, -111.947929],
                   [40.712312, -111.954971],
                   [40.774841, -111.885948],
                   [40.715443, -111.877740],
                   [40.654498, -111.957709],
                   [40.710026, -111.890679],
                   [40.775667, -111.888137],
                   [40.704667, -111.937086],
                   [40.702483, -111.922838],
                   [40.697871, -111.916820],
                   [40.703364, -111.909673],
                   [40.691036, -111.891020],
                   [40.708705, -111.901988],
                   [40.760331, -111.888332],
                   [40.676338, -111.854214],
                   [40.671442, -111.824646],
                   [40.662317, -111.887077],
                   [40.659050, -111.958002],
                   [40.653966, -111.865307],
                   [40.749703, -111.873752],
                   [40.635141, -111.864093]])

df = pd.DataFrame(
    {
        "lat": coords[:, 0],
        "lon": coords[:, 1],
        "col3": 20,
        "col4": "#f00",
    }
)

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=40.68,
            longitude=-111.90,
            zoom=11,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                get_radius=100,
            ),
            pdk.Layer(
            "LineLayer",
                df,
                get_source_position="[lon, lat]",
                target_position="[lon, lat]",
                get_color="[200, 30, 0, 160]",
                get_width=10,
                highlight_color=[255, 255, 0],
                picking_radius=10,
                auto_highlight=True,
                pickable=True,
            ),
        ],
    )
)