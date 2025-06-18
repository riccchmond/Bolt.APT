import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Configuration ---
CSV_FILEPATH = 'javelin_analysis_log.csv' # Path to the saved analysis data

# --- Data Loading ---
def load_analysis_data(csv_path):
    """Loads the analysis data from the CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: Analysis data file not found at {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        # Convert timestamp to datetime objects for proper sorting and filtering
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df)} records from {csv_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- Plotting Functions ---

# def plot_latest_throw_comparison(df, athlete_id, phase, angles_to_plot=None):
#     """
#     Generates a bar chart comparing calculated vs ideal angles for the latest
#     throw of a specific athlete and phase.
#     """
#     if df is None or df.empty:
#         print("No data available for plotting.")
#         return None
#
#     # Filter data for the specific athlete
#     df_athlete = df[df['athlete_id'] == athlete_id].copy()
#     if df_athlete.empty:
#         print(f"No data found for athlete {athlete_id}")
#         return None
#
#     # Find the latest analysis timestamp for this athlete
#     latest_timestamp = df_athlete['timestamp'].max()
#     df_latest = df_athlete[df_athlete['timestamp'] == latest_timestamp]
#
#     # Filter for the specific phase
#     df_phase = df_latest[df_latest['phase'] == phase].copy()
#     if df_phase.empty:
#         print(f"No data found for phase '{phase}' in the latest analysis for athlete {athlete_id}")
#         return None
#
#     # Filter for specific angles if provided
#     if angles_to_plot:
#         df_phase = df_phase[df_phase['angle_name'].isin(angles_to_plot)]
#         if df_phase.empty:
#             print(f"None of the specified angles {angles_to_plot} found for phase '{phase}'")
#             return None
#
#     # Prepare data for plotting (melting for grouped bar chart)
#     df_plot = df_phase[['angle_name', 'calculated', 'ideal']].melt(
#         id_vars='angle_name',
#         var_name='Angle Type',
#         value_name='Angle (Degrees)'
#     )
#
#     # Create the plot
#     fig = px.bar(df_plot,
#                  x='angle_name',
#                  y='Angle (Degrees)',
#                  color='Angle Type',
#                  barmode='group', # Group bars side-by-side
#                  title=f'Latest Throw Analysis: {phase.capitalize()} Phase ({athlete_id} @ {latest_timestamp.strftime("%Y-%m-%d %H:%M")})',
#                  labels={'angle_name': 'Joint Angle', 'Angle (Degrees)': 'Angle (°)'}
#                  )
#
#     fig.update_layout(xaxis_title="Joint Angle",
#                       yaxis_title="Angle (°)",
#                       legend_title="Angle Type")
#     return fig

def plot_latest_throw_comparison(df, athlete_id, phase, angles_to_plot=None):
    """
    Generates a line chart comparing calculated vs ideal angles for the latest
    throw of a specific athlete and phase.
    """
    if df is None or df.empty:
        print("No data available for plotting.")
        return None

    # Filter data for the specific athlete
    df_athlete = df[df['athlete_id'] == athlete_id].copy()
    if df_athlete.empty:
        print(f"No data found for athlete {athlete_id}")
        return None

    # Find the latest analysis timestamp for this athlete
    latest_timestamp = df_athlete['timestamp'].max()
    df_latest = df_athlete[df_athlete['timestamp'] == latest_timestamp]

    # Filter for the specific phase
    df_phase = df_latest[df_latest['phase'] == phase].copy()
    if df_phase.empty:
        print(f"No data found for phase '{phase}' in the latest analysis for athlete {athlete_id}")
        return None

    # Filter for specific angles if provided
    if angles_to_plot:
        df_phase = df_phase[df_phase['angle_name'].isin(angles_to_plot)]
        if df_phase.empty:
            print(f"None of the specified angles {angles_to_plot} found for phase '{phase}'")
            return None

    # Create figure with two lines
    fig = go.Figure()

    # Add calculated angles line
    fig.add_trace(go.Scatter(
        x=df_phase['angle_name'],
        y=df_phase['calculated'],
        mode='lines+markers',
        name='Calculated',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))

    # Add ideal angles line
    fig.add_trace(go.Scatter(
        x=df_phase['angle_name'],
        y=df_phase['ideal'],
        mode='lines+markers',
        name='Ideal',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=8)
    ))

    # Update layout
    fig.update_layout(
        title=f'Latest Throw Analysis: {phase.capitalize()} Phase<br>({athlete_id} @ {latest_timestamp.strftime("%Y-%m-%d %H:%M")})',
        xaxis_title="Joint Angle",
        yaxis_title="Angle (°)",
        legend_title="Angle Type",
        hovermode='x unified'
    )

    return fig

def plot_angle_trend(df, athlete_id, phase, angle_name):
    """
    Generates a line chart showing the trend of a specific calculated angle
    and its deviation over time for an athlete, phase, and angle.
    """
    if df is None or df.empty:
        print("No data available for plotting.")
        return None

    # Filter data
    df_filtered = df[
        (df['athlete_id'] == athlete_id) &
        (df['phase'] == phase) &
        (df['angle_name'] == angle_name)
    ].copy()

    if df_filtered.empty:
        print(f"No data found for athlete {athlete_id}, phase '{phase}', angle '{angle_name}'")
        return None

    # Sort by time
    df_filtered.sort_values('timestamp', inplace=True)

    # Create the plot with Plotly Graph Objects for dual-axis
    fig = go.Figure()

    # Add Calculated Angle trace
    fig.add_trace(go.Scatter(
        x=df_filtered['timestamp'],
        y=df_filtered['calculated'],
        mode='lines+markers',
        name='Calculated Angle (°)',
        yaxis='y1' # Assign to primary y-axis
    ))

    # Add Ideal Angle trace (optional, could be a straight line if constant, or vary if athlete profile changes)
    # For simplicity, let's assume it might vary slightly per analysis run if ideal changes
    fig.add_trace(go.Scatter(
        x=df_filtered['timestamp'],
        y=df_filtered['ideal'],
        mode='lines',
        name='Ideal Angle (°)',
        line=dict(dash='dash'),
        yaxis='y1'
    ))

    # Add Deviation trace on secondary y-axis
    fig.add_trace(go.Scatter(
        x=df_filtered['timestamp'],
        y=df_filtered['deviation'],
        mode='lines+markers',
        name='Deviation (°)',
        yaxis='y2' # Assign to secondary y-axis
    ))

    # Update layout for dual axes
    fig.update_layout(
        title=f'Trend for {angle_name.replace("_", " ").title()} ({phase.capitalize()} Phase) - {athlete_id}',
        xaxis_title='Date & Time of Analysis',
        yaxis=dict(
            title='Angle (°)',
            titlefont=dict(color="#1f77b4"),
            tickfont=dict(color="#1f77b4")
        ),
        yaxis2=dict(
            title='Deviation (°)',
            titlefont=dict(color="#ff7f0e"),
            tickfont=dict(color="#ff7f0e"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend_title="Metric"
    )

    return fig

# --- Main Execution Example ---
if __name__ == "__main__":
    # 1. Load the data
    analysis_data = load_analysis_data(CSV_FILEPATH)

    if analysis_data is not None:
        # --- Example Usage ---
        target_athlete = 'Athlete_001' # ID of the athlete to visualize
        target_phase = 'release'       # Phase to analyze
        target_angle = 'dominant_elbow'# Angle to see trend for
        # Optional: list of angles for the comparison chart
        angles_for_comparison = ['dominant_elbow', 'dominant_shoulder', 'left_knee', 'right_knee']

        # 2. Generate Comparison Plot for Latest Throw
        fig_comparison = plot_latest_throw_comparison(analysis_data,
                                                       target_athlete,
                                                       target_phase,
                                                       angles_to_plot=angles_for_comparison)
        if fig_comparison:
            # fig_comparison.show() # Show plot in browser/IDE window
            fig_comparison.write_html(f"{target_athlete}_{target_phase}_latest_comparison.html") # Save as HTML
            print(f"Saved latest throw comparison plot to {target_athlete}_{target_phase}_latest_comparison.html")


        # 3. Generate Trend Plot for a Specific Angle
        fig_trend = plot_angle_trend(analysis_data,
                                      target_athlete,
                                      target_phase,
                                      target_angle)
        if fig_trend:
            # fig_trend.show() # Show plot in browser/IDE window
            fig_trend.write_html(f"{target_athlete}_{target_phase}_{target_angle}_trend.html") # Save as HTML
            print(f"Saved angle trend plot to {target_athlete}_{target_phase}_{target_angle}_trend.html")

    else:
        print("Could not load data for visualization.")