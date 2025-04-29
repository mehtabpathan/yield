from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html


# 1. Generate Dummy Data
def generate_dummy_data(n_vars=10, n_points=365):
    """Generates a dummy time series DataFrame."""
    start_date = datetime.now() - timedelta(days=n_points - 1)
    dates = pd.date_range(start_date, periods=n_points, freq="D")
    data = {
        f"Var{i+1}": np.random.randn(n_points).cumsum() + 50 + i * 10
        for i in range(n_vars)
    }
    df = pd.DataFrame(data, index=dates)
    return df


df = generate_dummy_data()
df_display = df.reset_index().rename(
    columns={"index": "Date"}
)  # For display in tables if needed

# Initialize the app
# Add Google Font stylesheet link
google_fonts = (
    "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap"
)
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        google_fonts,
        dbc.icons.BOOTSTRAP,
    ],  # Added Google Font and Bootstrap Icons
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],  # Responsive meta tag
)
server = app.server  # Expose server for deployment


# --- Helper function for shared TS plot layouts ---
def shared_ts_layout(title=""):
    """Creates a shared layout dictionary for the top time series plots."""
    return {
        "template": "plotly_dark",
        "title": {"text": title, "x": 0.5, "xanchor": "center"},  # Center title
        "height": 140,  # Adjusted height for 6 plots
        "margin": dict(l=40, r=10, t=40, b=20),  # Reduced bottom margin slightly
        "xaxis_title": None,  # Remove x-axis title (shared context)
        "yaxis_title": "Value",
        "xaxis_showticklabels": True,  # Ensure ticks are shown
        "yaxis_showgrid": True,  # Show grid lines
        "xaxis_showgrid": True,
        "font": {"family": "Poppins, sans-serif"},  # Apply font
    }


# --- Helper function to create time series graphs ---
def create_ts_graph(graph_id, variable_name):
    """Creates a dcc.Graph component for a time series plot."""
    return dcc.Graph(
        id=graph_id,
        figure=go.Figure(layout=shared_ts_layout(title=variable_name)),
        config={"displaylogo": False},
        className="mb-1",  # Add small bottom margin between plots
    )


# --- Layout Definition ---
app.layout = dbc.Container(
    [
        # Title
        dbc.Row(
            dbc.Col(
                html.H1("Scenario Simulation Dashboard", className="text-center"),
                width=12,
            ),  # Centered Title
            className="mb-4 mt-4",
        ),
        # Top Section: Filters and Time Series Plots
        dbc.Row(
            [
                # Left Column: Controls
                dbc.Col(
                    dbc.Card(
                        [  # Wrap controls in a Card
                            dbc.CardHeader("Controls & Filtered Data"),
                            dbc.CardBody(
                                [
                                    html.H5(
                                        "Time Frame Selection", className="card-title"
                                    ),
                                    # html.P("Please select time frame for your dataset:"), # Redundant with H5 title
                                    dcc.DatePickerRange(
                                        id="date-range-selector",
                                        min_date_allowed=df.index.min().date(),
                                        max_date_allowed=df.index.max().date(),
                                        start_date=df.index.min().date(),
                                        end_date=df.index.max().date(),
                                        className="mb-3 d-block",  # d-block for full width
                                        display_format="YYYY-MM-DD",  # Standard format
                                    ),
                                    html.H5(
                                        "Constraint Selector",
                                        className="card-title mt-4",
                                    ),
                                    dash_table.DataTable(
                                        id="constraint-table",
                                        columns=[
                                            {
                                                "name": "Variable",
                                                "id": "Var Name",
                                                "editable": False,
                                            },  # Renamed for clarity
                                            {
                                                "name": "Min",
                                                "id": "Min",
                                                "type": "numeric",
                                                "editable": True,
                                            },
                                            {
                                                "name": "Max",
                                                "id": "Max",
                                                "type": "numeric",
                                                "editable": True,
                                            },
                                        ],
                                        data=[
                                            {"Var Name": col, "Min": None, "Max": None}
                                            for col in df.columns[:5]
                                        ],
                                        editable=True,
                                        style_table={"overflowX": "auto"},
                                        style_cell={  # Adjusted styles for dark theme
                                            "textAlign": "left",
                                            "padding": "8px",
                                            "backgroundColor": "#222",  # Dark background for cells
                                            "color": "#ddd",  # Light text
                                            "border": "1px solid #444",
                                            "fontFamily": "Poppins, sans-serif",  # Ensure font
                                        },
                                        style_header={
                                            "backgroundColor": "#333",  # Slightly lighter header
                                            "fontWeight": "bold",
                                            "color": "white",
                                            "border": "1px solid #555",
                                            "fontFamily": "Poppins, sans-serif",
                                        },
                                        style_data={  # Inherits from style_cell, can add specifics if needed
                                            "border": "1px solid #444",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"column_id": "Var Name"},
                                                "fontWeight": "bold",
                                                "backgroundColor": "#2a2a2a",
                                            },
                                            {  # Add hover effect
                                                "if": {
                                                    "state": "active"
                                                },  # 'active' is the hovered cell
                                                "backgroundColor": "#444",
                                                "border": "1px solid #777",
                                                "color": "white",
                                            },
                                        ],
                                    ),
                                    dbc.Button(
                                        [
                                            "Apply Filter ",
                                            html.I(
                                                className="bi bi-filter-circle-fill ml-1"
                                            ),
                                        ],  # Added icon
                                        id="apply-filter-button",
                                        color="warning",
                                        className="mt-3 mb-4 w-100",  # Full width button
                                        n_clicks=0,
                                    ),
                                    # Placeholder for Data Visualizer Plot
                                    # html.H5("Data Visualizer Based on Constraints and Date Range"), # Title integrated into plot
                                    dcc.Graph(
                                        id="data-visualizer-plot",
                                        figure=go.Figure(
                                            layout={
                                                "template": "plotly_dark",
                                                "xaxis": {"title": "Date"},
                                                "yaxis": {"title": "Value"},
                                                "title": "Filtered Data Overview",
                                                "font": {
                                                    "family": "Poppins, sans-serif"
                                                },  # Apply font to plot
                                                "height": 350,  # Adjusted height
                                                "margin": dict(l=40, r=20, t=50, b=40),
                                            }
                                        ),
                                        config={
                                            "displaylogo": False
                                        },  # Hide Plotly logo
                                    ),
                                ]
                            ),  # End CardBody
                        ]
                    ),  # End Card
                    md=5,
                    className="mb-4 mb-md-0",  # Add margin bottom on small screens
                ),  # End Left Column
                # Right Column: Time Series Plots
                dbc.Col(
                    dbc.Card(
                        [  # Wrap plots in a Card
                            dbc.CardHeader("Individual Time Series (Var 1-6)"),
                            dbc.CardBody(
                                [
                                    create_ts_graph("ts-plot-1", "Var1"),
                                    create_ts_graph("ts-plot-2", "Var2"),
                                    create_ts_graph("ts-plot-3", "Var3"),
                                    create_ts_graph("ts-plot-4", "Var4"),
                                    create_ts_graph("ts-plot-5", "Var5"),
                                    create_ts_graph("ts-plot-6", "Var6"),
                                ]
                            ),  # End CardBody
                        ]
                    ),  # End Card
                    md=7,
                ),  # End Right Column
            ],
            # className="mb-5", # Margin added by Card
        ),  # End Top Section Row
        # Bottom Section: Scenario Simulation
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [  # Wrap simulation section in a Card
                            dbc.CardHeader("Scenario Simulation"),
                            dbc.CardBody(
                                [
                                    # html.H5("Simulate Scenarios"), # Title in CardHeader
                                    dash_table.DataTable(
                                        id="scenario-table",
                                        columns=[
                                            {
                                                "name": "Scenario",
                                                "id": "Scenario",
                                                "editable": False,
                                            },
                                            {
                                                "name": "IJ1",
                                                "id": "IJ1",
                                                "type": "numeric",
                                                "editable": True,
                                            },
                                            {
                                                "name": "IJ2",
                                                "id": "IJ2",
                                                "type": "numeric",
                                                "editable": True,
                                            },
                                            {
                                                "name": "IJ3",
                                                "id": "IJ3",
                                                "type": "numeric",
                                                "editable": True,
                                            },
                                            {
                                                "name": "A",
                                                "id": "A",
                                                "type": "numeric",
                                                "editable": False,
                                            },
                                            {
                                                "name": "B",
                                                "id": "B",
                                                "type": "numeric",
                                                "editable": False,
                                            },
                                            {
                                                "name": "C",
                                                "id": "C",
                                                "type": "numeric",
                                                "editable": False,
                                            },
                                            {
                                                "name": "D",
                                                "id": "D",
                                                "type": "numeric",
                                                "editable": False,
                                            },
                                            {
                                                "name": "E",
                                                "id": "E",
                                                "type": "numeric",
                                                "editable": False,
                                            },
                                            {
                                                "name": "F",
                                                "id": "F",
                                                "type": "numeric",
                                                "editable": False,
                                            },
                                        ],
                                        data=[
                                            {
                                                "Scenario": "Base",
                                                "IJ1": 12,
                                                "IJ2": 323,
                                                "IJ3": 12,
                                                "A": 52,
                                                "B": 48,
                                                "C": 67,
                                                "D": 65,
                                                "E": 62,
                                                "F": 75,
                                            },
                                            {
                                                "Scenario": "Scenario 1",
                                                "IJ1": None,
                                                "IJ2": None,
                                                "IJ3": None,
                                                "A": None,
                                                "B": None,
                                                "C": None,
                                                "D": None,
                                                "E": None,
                                                "F": None,
                                            },
                                            {
                                                "Scenario": "Scenario 2",
                                                "IJ1": None,
                                                "IJ2": None,
                                                "IJ3": None,
                                                "A": None,
                                                "B": None,
                                                "C": None,
                                                "D": None,
                                                "E": None,
                                                "F": None,
                                            },
                                            {
                                                "Scenario": "Scenario 3",
                                                "IJ1": None,
                                                "IJ2": None,
                                                "IJ3": None,
                                                "A": None,
                                                "B": None,
                                                "C": None,
                                                "D": None,
                                                "E": None,
                                                "F": None,
                                            },
                                        ],
                                        editable=True,
                                        style_table={"overflowX": "auto"},
                                        style_cell={  # Use consistent dark styling
                                            "textAlign": "left",
                                            "padding": "8px",
                                            "backgroundColor": "#222",
                                            "color": "#ddd",
                                            "border": "1px solid #444",
                                            "fontFamily": "Poppins, sans-serif",
                                        },
                                        style_header={
                                            "backgroundColor": "#333",
                                            "fontWeight": "bold",
                                            "color": "white",
                                            "border": "1px solid #555",
                                            "fontFamily": "Poppins, sans-serif",
                                        },
                                        style_data={
                                            "border": "1px solid #444",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"column_id": "Scenario"},
                                                "fontWeight": "bold",
                                                "backgroundColor": "#2a2a2a",
                                            },
                                            {
                                                "if": {
                                                    "filter_query": '{Scenario} = "Base"',
                                                    "column_editable": True,
                                                },
                                                "editable": False,
                                                "backgroundColor": "#3a3a3a",
                                            },  # Darker non-editable base
                                            {
                                                "if": {
                                                    "filter_query": '{Scenario} != "Base"',
                                                    "column_id": [
                                                        "A",
                                                        "B",
                                                        "C",
                                                        "D",
                                                        "E",
                                                        "F",
                                                    ],
                                                },
                                                "editable": False,
                                                "backgroundColor": "#282828",
                                            },  # Slightly different bg for non-editable derived
                                            {  # Hover effect
                                                "if": {"state": "active"},
                                                "backgroundColor": "#444",
                                                "border": "1px solid #777",
                                                "color": "white",
                                            },
                                        ],
                                    ),
                                    dbc.Button(
                                        [
                                            "Predict ",
                                            html.I(
                                                className="bi bi-graph-up-arrow ml-1"
                                            ),
                                        ],  # Added icon
                                        id="predict-button",
                                        color="success",
                                        className="mt-3 mb-4 w-100",  # Full width
                                        n_clicks=0,
                                    ),
                                    # Placeholder for Scenario Plot
                                    # html.H5("Line Plot Based on Scenarios"), # Title in CardHeader / Plot
                                    dcc.Graph(
                                        id="scenario-plot",
                                        figure=go.Figure(
                                            layout={
                                                "template": "plotly_dark",
                                                "xaxis": {"title": "Date / Time"},
                                                "yaxis": {"title": "Predicted Value"},
                                                "title": "Scenario Simulation Results",
                                                "font": {
                                                    "family": "Poppins, sans-serif"
                                                },  # Apply font
                                                "height": 400,  # Adjusted height
                                                "margin": dict(l=40, r=20, t=50, b=40),
                                            }
                                        ),
                                        config={"displaylogo": False},
                                    ),
                                ]
                            ),  # End CardBody
                        ]
                    ),  # End Card
                    width=12,
                )  # End Column
            ]
        ),  # End Bottom Section Row
        # Stores for intermediate data (no visual change)
        dcc.Store(id="store-df-filtered"),
        dcc.Store(id="store-df-constrained"),
        dcc.Store(id="store-scenario-data"),
    ],
    fluid=True,  # Use fluid container for full width
    className="dbc",  # Apply dbc class for better Bootstrap integration if needed
)


# --- Callbacks ---


# Callback 1 & 2: Date Range -> Filtered Data Store -> Update Time Series Plots
@callback(
    Output("store-df-filtered", "data"),
    Input("date-range-selector", "start_date"),
    Input("date-range-selector", "end_date"),
)
def update_filtered_data_store(start_date, end_date):
    if start_date and end_date:
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_filtered = df.loc[mask]
        return df_filtered.to_json(date_format="iso", orient="split")
    return dash.no_update  # Or return empty df JSON


# Updated Callbacks for TS Plots 1-3 to use the helper layout function
@callback(Output("ts-plot-1", "figure"), Input("store-df-filtered", "data"))
def update_ts_plot_1(jsonified_filtered_data):
    layout = shared_ts_layout(title="Var1")  # Use helper
    if not jsonified_filtered_data:
        return go.Figure(layout=layout)

    dff = pd.read_json(jsonified_filtered_data, orient="split")
    fig = go.Figure(layout=layout)  # Apply layout on creation
    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["Var1"],
            mode="lines",
            name="Var1",
            line=dict(color="#636EFA"),
        )
    )  # Specific color
    fig.update_layout(xaxis_rangeslider_visible=True)  # Keep slider only on plot 1
    return fig


@callback(Output("ts-plot-2", "figure"), Input("store-df-filtered", "data"))
def update_ts_plot_2(jsonified_filtered_data):
    layout = shared_ts_layout(title="Var2")  # Use helper
    if not jsonified_filtered_data:
        return go.Figure(layout=layout)

    dff = pd.read_json(jsonified_filtered_data, orient="split")
    fig = go.Figure(layout=layout)  # Apply layout
    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["Var2"],
            mode="lines",
            name="Var2",
            line=dict(color="#FFA15A"),
        )
    )  # Specific color (orange)
    fig.update_layout(xaxis_rangeslider_visible=False)  # Ensure no slider
    return fig


@callback(Output("ts-plot-3", "figure"), Input("store-df-filtered", "data"))
def update_ts_plot_3(jsonified_filtered_data):
    layout = shared_ts_layout(title="Var3")  # Use helper
    if not jsonified_filtered_data:
        return go.Figure(layout=layout)

    dff = pd.read_json(jsonified_filtered_data, orient="split")
    fig = go.Figure(layout=layout)  # Apply layout
    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["Var3"],
            mode="lines",
            name="Var3",
            line=dict(color="#00CC96"),
        )
    )  # Specific color (green)
    fig.update_layout(xaxis_rangeslider_visible=False)  # Ensure no slider
    return fig


# --- Callbacks for Plots 4-6 ---
@callback(Output("ts-plot-4", "figure"), Input("store-df-filtered", "data"))
def update_ts_plot_4(jsonified_filtered_data):
    layout = shared_ts_layout(title="Var4")  # Use helper
    if not jsonified_filtered_data:
        return go.Figure(layout=layout)

    dff = pd.read_json(jsonified_filtered_data, orient="split")
    # Check if Var4 exists in the potentially filtered dataframe
    if "Var4" not in dff.columns:
        return go.Figure(layout=layout.update(title="Var4 (Not in selected data)"))

    fig = go.Figure(layout=layout)  # Apply layout
    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["Var4"],
            mode="lines",
            name="Var4",
            line=dict(color="#EF553B"),  # Red
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False)  # Ensure no slider
    return fig


@callback(Output("ts-plot-5", "figure"), Input("store-df-filtered", "data"))
def update_ts_plot_5(jsonified_filtered_data):
    layout = shared_ts_layout(title="Var5")  # Use helper
    if not jsonified_filtered_data:
        return go.Figure(layout=layout)

    dff = pd.read_json(jsonified_filtered_data, orient="split")
    if "Var5" not in dff.columns:
        return go.Figure(layout=layout.update(title="Var5 (Not in selected data)"))

    fig = go.Figure(layout=layout)  # Apply layout
    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["Var5"],
            mode="lines",
            name="Var5",
            line=dict(color="#AB63FA"),  # Purple
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False)  # Ensure no slider
    return fig


@callback(Output("ts-plot-6", "figure"), Input("store-df-filtered", "data"))
def update_ts_plot_6(jsonified_filtered_data):
    layout = shared_ts_layout(title="Var6")  # Use helper
    if not jsonified_filtered_data:
        return go.Figure(layout=layout)

    dff = pd.read_json(jsonified_filtered_data, orient="split")
    if "Var6" not in dff.columns:
        return go.Figure(layout=layout.update(title="Var6 (Not in selected data)"))

    fig = go.Figure(layout=layout)  # Apply layout
    fig.add_trace(
        go.Scatter(
            x=dff.index,
            y=dff["Var6"],
            mode="lines",
            name="Var6",
            line=dict(color="#FECB52"),  # Yellow
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False)  # Ensure no slider
    return fig


# Callback to synchronize x-axis zoom/pan/range slider across the six time series plots
@callback(
    Output("ts-plot-2", "figure", allow_duplicate=True),
    Output("ts-plot-3", "figure", allow_duplicate=True),
    Output("ts-plot-4", "figure", allow_duplicate=True),
    Output("ts-plot-5", "figure", allow_duplicate=True),
    Output("ts-plot-6", "figure", allow_duplicate=True),
    Input("ts-plot-1", "relayoutData"),
    State("ts-plot-2", "figure"),
    State("ts-plot-3", "figure"),
    State("ts-plot-4", "figure"),
    State("ts-plot-5", "figure"),
    State("ts-plot-6", "figure"),
    prevent_initial_call=True,
)
def sync_xaxis_ranges(relayout_data, fig2, fig3, fig4, fig5, fig6):
    """Updates the xaxis range of plots 2-6 based on plot 1's relayoutData."""
    if not relayout_data:
        return (dash.no_update,) * 5

    updated_fig2 = go.Figure(fig2)
    updated_fig3 = go.Figure(fig3)
    updated_fig4 = go.Figure(fig4)
    updated_fig5 = go.Figure(fig5)
    updated_fig6 = go.Figure(fig6)

    figures_to_update = [
        updated_fig2,
        updated_fig3,
        updated_fig4,
        updated_fig5,
        updated_fig6,
    ]

    range_updated = False
    x_range = None

    # Check common keys for x-axis range changes from relayoutData
    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        # Handles zoom, pan, potentially slider depending on version/interaction
        x_range = [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]
        range_updated = True
    elif (
        "xaxis.range" in relayout_data
        and isinstance(relayout_data["xaxis.range"], list)
        and len(relayout_data["xaxis.range"]) == 2
    ):
        # Handles range slider in some cases
        x_range = relayout_data["xaxis.range"]
        range_updated = True
    elif "xaxis.autorange" in relayout_data and relayout_data["xaxis.autorange"]:
        # Handles double-click to reset axis
        for fig in figures_to_update:
            fig.update_layout(xaxis_range=None, xaxis_autorange=True)
        # Return directly after setting autorange for all
        return updated_fig2, updated_fig3, updated_fig4, updated_fig5, updated_fig6

    # Apply the determined range if an update occurred
    if range_updated and x_range is not None:
        # Ensure the range values are appropriate (strings for dates, numbers for numeric)
        # Plotly usually handles this, but basic check can be useful
        # print(f"Applying range: {x_range}") # Uncomment for debugging
        for fig in figures_to_update:
            fig.update_layout(xaxis_range=x_range, xaxis_autorange=None)
        return updated_fig2, updated_fig3, updated_fig4, updated_fig5, updated_fig6
    else:
        # No relevant range update information found
        return (dash.no_update,) * 5


# Callback 3 & 4: Apply Filter Button -> Update Constrained Data Store -> Update Data Visualizer Plot
@callback(
    Output("store-df-constrained", "data"),
    Input("apply-filter-button", "n_clicks"),
    State("store-df-filtered", "data"),
    State("constraint-table", "data"),
)
def apply_constraints_and_update_store(n_clicks, jsonified_filtered_data, constraints):
    if n_clicks == 0 or not jsonified_filtered_data:
        return dash.no_update  # Or return the filtered data if no constraints

    dff = pd.read_json(jsonified_filtered_data, orient="split")
    df_constrained = dff.copy()  # Start with the date-filtered data

    for constraint in constraints:
        var = constraint.get("Var Name")
        min_val = constraint.get("Min")
        max_val = constraint.get("Max")

        if var and var in df_constrained.columns:
            try:
                # Apply min constraint if specified and valid
                if min_val is not None:
                    min_val = float(min_val)
                    df_constrained = df_constrained[df_constrained[var] >= min_val]
                # Apply max constraint if specified and valid
                if max_val is not None:
                    max_val = float(max_val)
                    df_constrained = df_constrained[df_constrained[var] <= max_val]
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid numeric constraint for {var}. Min: {min_val}, Max: {max_val}"
                )
                # Handle error or skip constraint application for this variable
                pass  # Skip this constraint if conversion fails

    # Note: Implementing nearest neighbour if constraints are not met would require more complex logic here.
    # For now, it strictly filters based on the provided min/max.

    return df_constrained.to_json(date_format="iso", orient="split")


@callback(
    Output("data-visualizer-plot", "figure"),
    Input("store-df-constrained", "data"),  # Triggered when constrained data updates
)
def update_data_visualizer(jsonified_constrained_data):
    layout = {  # Define layout consistent with others
        "template": "plotly_dark",
        "xaxis": {"title": "Date"},
        "yaxis": {"title": "Value"},
        "title": "Filtered Data Overview",
        "font": {"family": "Poppins, sans-serif"},
        "height": 350,
        "margin": dict(l=40, r=20, t=50, b=40),
    }
    if not jsonified_constrained_data:
        layout["title"] = "Filtered Data Overview (No data matching constraints)"
        return go.Figure(layout=layout)

    dff_constrained = pd.read_json(jsonified_constrained_data, orient="split")
    fig = go.Figure(layout=layout)  # Apply layout

    # Plot all variables present in the constrained data using scatter (markers)
    for col in dff_constrained.columns:
        fig.add_trace(
            go.Scatter(
                x=dff_constrained.index,
                y=dff_constrained[col],
                mode="markers",  # <-- Changed to markers
                name=col,
                marker=dict(size=5, opacity=0.7),  # Style markers
            )
        )
    return fig


# Callback 5: Predict Button -> Save Scenario Data to Store
@callback(
    Output("store-scenario-data", "data"),
    Input("predict-button", "n_clicks"),
    State("scenario-table", "data"),
    prevent_initial_call=True,  # Don't run on app load
)
def save_scenario_data(n_clicks, scenario_data):
    if n_clicks > 0 and scenario_data:
        # Convert to DataFrame for easier handling later, maybe store as JSON dict list
        # For now, just pass the raw list of dicts from the table
        return scenario_data
    return dash.no_update


# Callback 6: Scenario Data Store Update -> Update Scenario Plot
@callback(
    Output("scenario-plot", "figure"),
    Input("store-scenario-data", "data"),  # Triggered by predict button callback
    State("store-df-constrained", "data"),  # Get the potentially constrained data
    prevent_initial_call=True,  # Don't run on app load
)
def update_scenario_plot(scenario_data, jsonified_constrained_data):
    layout = {  # Define layout consistent with others
        "template": "plotly_dark",
        "xaxis": {"title": "Date"},  # Assuming time-based plot for now
        "yaxis": {"title": "Simulated Output"},
        "title": "Scenario Simulation Results",
        "font": {"family": "Poppins, sans-serif"},
        "height": 400,
        "margin": dict(l=40, r=20, t=50, b=40),
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.2,
            "xanchor": "center",
            "x": 0.5,
        },  # Move legend below
    }

    if not scenario_data:
        return go.Figure(layout=layout)  # Return empty fig with layout

    # Base data to use for scenarios
    if jsonified_constrained_data:
        base_df = pd.read_json(jsonified_constrained_data, orient="split")
        if base_df.empty:  # If constraints resulted in empty df
            layout["title"] = (
                "Scenario Simulation Results (Base data empty after filtering)"
            )
            return go.Figure(layout=layout)
    else:
        # Fallback if no constrained data exists (e.g., filter not applied yet)
        # Depending on desired behavior, could use original df or show message
        # Let's use original df for base plotting if constrained is unavailable
        base_df = (
            df.copy()
        )  # Use original df as fallback - check date range? Maybe filter by date selector?
        start_date = df.index.min().date()  # Get default range if needed
        end_date = df.index.max().date()
        # This part might need refinement based on exact desired fallback behavior
        print(
            "Warning: Constrained data not available for scenario plot, using original df as base."
        )
        # Consider filtering base_df by date_range_selector values here if needed

    fig = go.Figure(layout=layout)  # Apply layout

    # Plot Base line (only if Var1 exists)
    if not base_df.empty and "Var1" in base_df.columns:
        fig.add_trace(
            go.Scatter(
                x=base_df.index,
                y=base_df["Var1"],
                mode="lines",
                name="Base (Var1)",
                line=dict(dash="dash", color="#888"),  # Dashed grey line
            )
        )

    # Plot Scenarios
    color_map = plotly.colors.qualitative.Plotly  # Use a qualitative color scale

    for i, scenario in enumerate(scenario_data):
        if scenario["Scenario"] != "Base":
            ij1 = scenario.get("IJ1")
            ij2 = scenario.get("IJ2")
            ij3 = scenario.get("IJ3")
            name = scenario["Scenario"]
            color = color_map[i % len(color_map)]  # Cycle through colors

            if not base_df.empty and "Var1" in base_df.columns:
                simulated_y = base_df["Var1"].copy()
                try:
                    if ij1 is not None:
                        simulated_y += float(ij1) * 0.1
                    if ij2 is not None:
                        simulated_y += float(ij2) * 0.01
                    if ij3 is not None:
                        simulated_y += float(ij3) * 0.5

                    fig.add_trace(
                        go.Scatter(
                            x=base_df.index,
                            y=simulated_y,
                            mode="markers",  # <-- Changed to markers
                            name=name,
                            marker=dict(
                                size=6, opacity=0.8, color=color
                            ),  # Style markers
                        )
                    )
                except (ValueError, TypeError):
                    print(f"Warning: Invalid numeric input for {name}. Skipping plot.")

    return fig


# Run the app
if __name__ == "__main__":
    app.run(debug=True)  # Turn off debug=True for production
