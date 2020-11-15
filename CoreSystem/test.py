import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots   
import pandas as pd

# read in volcano database data
df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/volcano_db.csv",
    encoding="iso-8859-1",
)

# frequency of Country
freq = df
freq = freq.Country.value_counts().reset_index().rename(columns={"index": "x"})

# read in 3d volcano surface data
df_v = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/volcano.csv")

# Initialize figure with subplots
fig = make_subplots(
    rows=3, cols=2,
    column_widths=[0.6, 0.4],
    row_heights=[0.4, 0.4, 0.6],
    specs=[[{"type": "scattergeo", "rowspan": 2}, {"type": "scatter"}],
           [        None          , {"type": "scatter"}],
           [{"type": "mesh3d"}, {"type": "scattermapbox"}]]
)
#"rowspan": 2

# Add scattergeo globe map of volcano locations
fig.add_trace(
    go.Scattergeo(lat=df["Latitude"],
                  lon=df["Longitude"],
                  mode="markers",
                  hoverinfo="text",
                  showlegend=False,
                  marker=dict(color="crimson", size=4, opacity=0.8)),
    row=1, col=1
)

# Add locations bar chart
fig.add_trace(
    go.Scatter(x=freq["x"][0:10],y=freq["Country"][0:10], marker=dict(color="crimson"), showlegend=False),
    row=1, col=2
)


# Add locations bar chart
fig.add_trace(
    go.Scatter(x=freq["x"][0:10],y=freq["Country"][0:10], marker=dict(color="crimson"), showlegend=False),
    row=2, col=2
)

fig_aux = px.line_3d(df, x="Latitude", y="Longitude", z="Longitude")
fig.append_trace(fig_aux['data'][0],3,1)
fig.add_trace(go.Mesh3d(x=df["Latitude"],
                    y=df["Longitude"],
                    z=df["Longitude"],
                    color='rgba(108, 122, 137, 1)',
                    colorbar_title='z',
                    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                    name='y',
                    showscale=True
                    ))

fig.add_trace(go.Scattermapbox(
                    fill = "toself",
                    lon = [-74, -70, -70, -74], lat = [47, 47, 45, 45],
                    marker = { 'size': 10, 'color': "orange" }),
                    row=3, col=2)

fig.add_trace(go.Scattermapbox(
    mode = "markers+lines",
    lon = [-70, -72,-74],
    lat = [45, 46, 47],
    marker = {'size': 10}))

fig.update_layout(
    mapbox = {
        'style': "stamen-terrain",
        'center': {'lon': -73, 'lat': 46 },
        'zoom': 5},
    showlegend = False)
# Add 3d surface of volcano
#fig.add_trace(
#    go.Surface(z=df_v.values.tolist(), showscale=False),
#    row=3, col=2
#)

# Update geo subplot properties
fig.update_geos(
    projection_type="orthographic",
    landcolor="white",
    oceancolor="MidnightBlue",
    showocean=True,
    lakecolor="LightBlue"
)

# Rotate x-axis labels
fig.update_xaxes(tickangle=45)

# Set theme, margin, and annotation in layout
fig.update_layout(
    template="plotly_dark",
    margin=dict(r=10, t=25, b=40, l=60),
    annotations=[
        dict(
            text="Source: NOAA",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=0)
    ]
)

fig.show()