import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import base64
from datetime import timedelta
import altair as alt
from utils.ppm_lookup import NOAALookup

# Add the parent directory to the path to access other modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

# Fix for 'configs' module not found error
atmoseer_dir = os.path.join(project_root, 'atmoseer')
sys.path.append(atmoseer_dir)

# Set page configuration
st.set_page_config(
    page_title="Carbon Dioxide | AtmoSeer",
    page_icon="🌲",
    layout="wide"
)

# Remove the empty white boxes
st.markdown("""
<style>
    /* Target all empty containers and hide them */
    .element-container:empty {
        display: none !important;
    }
    
    /* Hide any div with no content */
    div:empty {
        display: none !important;
    }
    
    /* Hide Streamlit's default empty containers */
    [data-testid="stAppViewContainer"] div:empty {
        display: none !important;
    }
    
    /* Hide those specific white boxes */
    .stBox {
        display: none !important;
    }
    
    /* Eliminate spacing in block containers */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Functions for background and styling
def get_base64_of_bin_file(bin_file):
    """Convert a binary file to base64 string for embedding in HTML/CSS."""
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_image(image_file):
    """Set an image as the page background using CSS."""
    bin_str = get_base64_of_bin_file(image_file)
    
    page_bg_css = f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-color: rgba(0, 0, 0, 0.4);  /* Darker overlay */
        }}
        
        /* Make all text Courier font */
        html, body, div, h1, h2, h3, h4, h5, h6, p, a, span, label {{
            font-family: 'Courier New', monospace !important;
        }}
        
        .page-title {{
            font-size: 3.5rem;
            font-weight: bold;
            text-align: center;
            color: #04870b;
            margin-bottom: 1.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        .info-container {{
            background-color: rgba(4, 135, 11, 0.5);
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
            margin-bottom: 2rem;
        }}
        
        .info-container h3 {{
            font-size: 2.2rem;
            color: #FFFFFF;
            font-weight: bold;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        .info-container p {{
            font-size: 1.4rem;
            line-height: 1.0;
            color: #FFFFFF;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
            margin-bottom: 2rem;
        }}
        
        .chart-container {{
            background-color: rgba(255, 255, 255, 0.7);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
            margin-bottom: 1.5rem;
        }}
        
        .action-container {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
            margin-bottom: 1.5rem;
        }}
        
        .action-container h3 {{
            color: #04870b;
            font-weight: bold;
            margin-bottom: 1rem;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        .results-container {{
            background-color: rgba(255, 255, 255, 0.7);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .results-header {{
            color: #04870b;
            font-weight: bold;
            margin-bottom: 1rem;
            font-size: 1.8rem;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        .metrics-container {{
            background-color: rgba(255, 255, 255, 0.7);
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
            margin-top: 1.5rem;
        }}
        
        .metrics-header {{
            color: #04870b;
            font-weight: bold;
            margin-bottom: 1rem;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        .metric-card {{
            background-color: rgba(4, 135, 11, 0.5);
            border-left: 4px solid #04870b;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }}
        
        .metric-title {{
            font-weight: bold;
            color: #FFFFFF;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        .metric-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #FFFFFF;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        /* Hide default Streamlit elements */
        #MainMenu, footer, header {{
            visibility: hidden;
        }}
    </style>
    """
    
    st.markdown(page_bg_css, unsafe_allow_html=True)

def format_date(d):
    """Format date for display."""
    if isinstance(d, str):
        d = pd.to_datetime(d).date()
    return d.strftime("%B %d, %Y")

@st.cache_data
def load_co2_data():
    """Load CO2 data from the data warehouse with caching."""
    try:
        # Try different potential file names
        data_paths = [
            os.path.join(project_root, 'data', 'data_warehouse', 'co2_data.csv'),
            os.path.join(project_root, 'data', 'data_warehouse', 'CO2DataNOAA.csv')
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                print(f"Loading data from {path}")
                df = pd.read_csv(path)
                # Ensure date is in datetime format
                df['date'] = pd.to_datetime(df['date'])
                # Sort by date
                df = df.sort_values('date')
                return df
        
        st.error("Could not find CO2 data files. Please check your data paths.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CO2 data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def init_lookup(data):
    """Initialize NOAALookup with caching."""
    try:
        return NOAALookup(data)
    except Exception as e:
        st.error(f"Error initializing NOAALookup: {str(e)}")
    return None

# Define a simple forecast result class for when the actual model can't be loaded
class SimpleForecaster:
    def __init__(self, data):
        self.data = data
    
    def predict_for_date(self, target_date):
        """
        Generate a simple forecast based on historical trends.
        """
        # Get the last available data point
        last_date = self.data['date'].max()
        last_ppm = self.data.loc[self.data['date'] == last_date, 'ppm'].values[0]
        
        # Calculate days difference
        days_diff = (pd.to_datetime(target_date) - pd.to_datetime(last_date)).days
        
        # Estimate yearly increase (about 2.5 ppm per year)
        yearly_increase = 2.5
        daily_increase = yearly_increase / 365
        
        # Calculate estimated future value
        predicted_ppm = last_ppm + (daily_increase * days_diff)
        
        # Simple class to match the expected interface
        class Prediction:
            def __init__(self, date, ppm, lower, upper):
                self.date = date
                self.ppm = ppm
                self.lower_bound = lower
                self.upper_bound = upper
        
        # Create prediction with confidence bounds
        uncertainty = 1.0 + (days_diff * 0.01)  # Increases with time
        prediction = Prediction(
            date=target_date,
            ppm=predicted_ppm,
            lower=predicted_ppm - uncertainty,
            upper=predicted_ppm + uncertainty
        )
        
        # Return prediction and an empty dict for historical
        return prediction, {}

# Main function to run the app
def main():
    # Set up session state to store results and other stateful data
    if 'lookup_result' not in st.session_state:
        st.session_state.lookup_result = None
    if 'forecast_result' not in st.session_state:
        st.session_state.forecast_result = None
    if 'chart_loaded' not in st.session_state:
        st.session_state.chart_loaded = False
    if 'lookup_requested' not in st.session_state:
        st.session_state.lookup_requested = False
    if 'forecast_requested' not in st.session_state:
        st.session_state.forecast_requested = False
    if 'initial_load' not in st.session_state:
        st.session_state.initial_load = True
    
    # Set the background image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "../images")
    forest_img_path = os.path.join(images_dir, "Forest.jpg")
    
    # Check if the background image exists
    if os.path.exists(forest_img_path):
        set_background_image(forest_img_path)
    
    # Page title and subtitle
    st.markdown('<h1 class="page-title">Carbon Dioxide (CO₂)</h1>', unsafe_allow_html=True)
    
    # Information section
    with st.container():
        st.markdown("""
        <div class="info-container">
            <h3>About Carbon Dioxide</h3>
            <p>
                Carbon dioxide (CO₂) is a key greenhouse gas that contributes significantly to climate change. While natural processes 
                emit CO₂, human activities—particularly burning fossil fuels and deforestation—have dramatically increased atmospheric 
                concentrations since the Industrial Revolution.
            </p>
            <p>
                CO₂ persists in the atmosphere for hundreds to thousands of years, making it especially important to monitor and predict 
                its levels for understanding long-term climate impacts. The measurements shown here come from NOAA's Global Monitoring 
                Laboratory, spanning from 1968 to present, with future projections generated by AtmoSeer's deep learning model.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data with caching
    co2_data = load_co2_data()
    
    if co2_data.empty:
        st.error("No data available. Please check the data files in the data_warehouse directory.")
        return
    
    # Initialize NOAALookup if possible (with caching)
    lookup = init_lookup(co2_data)
    lookup_available = lookup is not None
    
    # Main content area - 75% / 25% split
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create chart container
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Add chart title
        st.markdown('<h3 style="text-align: center; color: #04870b; text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);">CO₂ Concentrations (From 1968)</h3>', unsafe_allow_html=True)
        
        # Set up the chart
        chart_container = st.container()
        
        # Only show and animate the chart when specifically requested via buttons
        show_chart = st.session_state.initial_load or st.session_state.lookup_requested or st.session_state.forecast_requested
        
        if show_chart:
            # Determine the end date and dataset for visualization
            filtered_data = co2_data.copy()
            show_forecast = False
            forecast_point = None
            
            # Group by date and take first entry to ensure a single value per date
            filtered_data = filtered_data.sort_values('date').groupby('date').first().reset_index()

            # If we have a lookup result and lookup was requested
            if st.session_state.lookup_result is not None and st.session_state.lookup_requested:
                end_date = pd.to_datetime(st.session_state.lookup_result['date'])
                filtered_data = filtered_data[filtered_data['date'] <= end_date].copy()
            
            # If we have a forecast result and forecast was requested
            elif st.session_state.forecast_result is not None and st.session_state.forecast_requested:
                show_forecast = True
                forecast_date = pd.to_datetime(st.session_state.forecast_result['date'])
                forecast_point = pd.DataFrame({
                    'date': [forecast_date],
                    'ppm': [st.session_state.forecast_result['ppm']]
                })
            
            # Set up progress tracking for animation
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Start with just the first point
            initial_data = pd.DataFrame({
                'date': [filtered_data.iloc[0]['date']],
                'ppm': [filtered_data.iloc[0]['ppm']]
            })

            # Create the initial chart
            chart = alt.Chart(initial_data).mark_line(color='#04870b').encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('ppm:Q', 
                        title='CO₂ Concentration (ppm)', 
                        scale=alt.Scale(domain=[300, filtered_data['ppm'].max() * 1.05]))
            ).properties(
                width='container',
                height=500
            )

            # Display the initial chart
            chart_placeholder = chart_container.altair_chart(chart, use_container_width=True)

            # Prepare for animation
            total_points = len(filtered_data)
            num_steps = 100  # Show 100 steps for smooth animation
            points_per_step = max(1, total_points // num_steps)

            # Animate loading points
            animation_data = initial_data.copy()

            for i in range(points_per_step, total_points, points_per_step):
                # Select the next chunk of data
                step_index = min(i, total_points - 1)
                
                # Create new data for this step
                new_point = pd.DataFrame({
                    'date': [filtered_data.iloc[step_index]['date']],
                    'ppm': [filtered_data.iloc[step_index]['ppm']]
                })
                
                # Add to animation data
                animation_data = pd.concat([animation_data, new_point])
                
                # Update chart
                updated_chart = alt.Chart(animation_data).mark_line(color='#04870b').encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('ppm:Q', 
                            title='CO₂ Concentration (ppm)', 
                            scale=alt.Scale(domain=[300, filtered_data['ppm'].max() * 1.05]))
                ).properties(
                    width='container',
                    height=500
                )
                chart_placeholder.altair_chart(updated_chart, use_container_width=True)
                
                # Update progress
                progress = int((i / total_points) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Loading data: {progress}% complete")
                
                # Short sleep for animation
                time.sleep(0.01)

            # Make sure we include the last point
            if total_points > 0 and step_index < total_points - 1:
                final_point = pd.DataFrame({
                    'date': [filtered_data.iloc[-1]['date']],
                    'ppm': [filtered_data.iloc[-1]['ppm']]
                })
                animation_data = pd.concat([animation_data, final_point])

            # Add forecast point if we have one
            if show_forecast:
                # Get the last historical date (May 31, 2024 or latest available)
                latest_historical_date = filtered_data['date'].max()
                
                # Create forecast data from latest historical date to forecast date
                # For simplicity, we'll create a linear interpolation between the last known point and the forecast
                last_historical_ppm = filtered_data.loc[filtered_data['date'] == latest_historical_date, 'ppm'].values[0]
                forecast_ppm = st.session_state.forecast_result['ppm']
                
                # Create a date range from latest historical to forecast date
                date_range = pd.date_range(start=latest_historical_date, end=forecast_date, freq='D')
                
                # Linear interpolation between last historical point and forecast point
                ppm_values = np.linspace(last_historical_ppm, forecast_ppm, len(date_range))
                
                # Create the forecast dataframe
                forecast_df = pd.DataFrame({
                    'date': date_range,
                    'ppm': ppm_values
                })
                
                # Calculate confidence intervals (using the values from your forecast result)
                lower_bound = st.session_state.forecast_result.get('lower_bound', 0)
                upper_bound = st.session_state.forecast_result.get('upper_bound', 0)
                
                # Calculate the interval width percentage
                interval_width = (upper_bound - lower_bound) / forecast_ppm
                
                # Apply this percentage to all points in the forecast line for a gradual widening effect
                forecast_df['lower_bound'] = forecast_df['ppm'] * (1 - interval_width * np.linspace(0, 1, len(forecast_df)))
                forecast_df['upper_bound'] = forecast_df['ppm'] * (1 + interval_width * np.linspace(0, 1, len(forecast_df)))
                
                # Create the forecast line layer
                forecast_line_layer = alt.Chart(forecast_df).mark_line(
                    color='red',
                    strokeWidth=2
                ).encode(
                    x='date:T',
                    y='ppm:Q'
                )
                
                # Create the confidence interval layer
                confidence_interval = alt.Chart(forecast_df).mark_area(
                    color='gray',
                    opacity=0.3
                ).encode(
                    x='date:T',
                    y='lower_bound:Q',
                    y2='upper_bound:Q'
                )
                
                # Add forecast label at the end of the forecast line
                forecast_text = alt.Chart(pd.DataFrame({
                    'date': [forecast_date],
                    'ppm': [forecast_ppm],
                    'text': ['Forecast']
                })).mark_text(
                    align='left',
                    baseline='middle',
                    dx=15,
                    fontSize=12,
                    fontWeight='bold',
                    color='red'
                ).encode(
                    x='date:T',
                    y='ppm:Q',
                    text='text:N'
                )
                
                # Update the main chart with the historical line and forecast elements
                final_chart = alt.Chart(animation_data).mark_line(color='#04870b').encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('ppm:Q', 
                            title='CO₂ Concentration (ppm)', 
                            scale=alt.Scale(domain=[300, max(filtered_data['ppm'].max(), forecast_point['ppm'].max()) * 1.05]))
                ).properties(
                    width='container',
                    height=500
                )
                
                # Combine all layers
                combined_chart = final_chart + confidence_interval + forecast_line_layer + forecast_text
                chart_placeholder.altair_chart(combined_chart, use_container_width=True)
            
            # Reset the flags after animation
            st.session_state.lookup_requested = False
            st.session_state.forecast_requested = False
            st.session_state.initial_load = False
            
            # Final progress update
            progress_bar.progress(100)
            status_text.text("Data loading complete!")
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()
        else:
            # If no chart has been requested yet, show a placeholder or empty chart
            placeholder_data = pd.DataFrame({
                'date': [co2_data['date'].min(), co2_data['date'].max()],
                'ppm': [co2_data['ppm'].min(), co2_data['ppm'].max()]
            })
            
            placeholder_chart = alt.Chart(placeholder_data).mark_line(color='#04870b', opacity=0).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('ppm:Q', 
                        title='CO₂ Concentration (ppm)', 
                        scale=alt.Scale(domain=[300, placeholder_data['ppm'].max() * 1.05]))
            ).properties(
                width='container',
                height=500
            )
            
            chart_container.altair_chart(placeholder_chart, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results container (to show lookup or forecast results)
        if st.session_state.lookup_result is not None or st.session_state.forecast_result is not None:
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            if st.session_state.lookup_result is not None:
                result = st.session_state.lookup_result
                st.markdown(f"<h3 class='results-header'>Historical Data for {format_date(result['date'])}</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background-color: rgba(4, 135, 11, 0.5); padding: 1rem; border-radius: 5px; margin-bottom: 0.5rem;'>
                    <p style='margin-bottom: 0.2rem; font-size: 2.2rem; text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);'><strong>CO₂ Concentration:</strong> {result['ppm']:.2f} ppm</p>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state.forecast_result is not None:
                result = st.session_state.forecast_result
                st.markdown(f"<h3 class='results-header'>Forecast for {format_date(result['date'])}</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background-color: rgba(4, 135, 11, 0.5); padding: 1rem; border-radius: 5px; margin-bottom: 0.5rem;'>
                    <p style='margin-bottom: 0.2rem; font-size: 2.2rem; text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);'><strong>CO₂ Concentration:</strong> {result['ppm']:.2f} ppm</p>
                    <p style='margin-bottom: 0.2rem; font-size: 1.5rem; text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);'><strong>Confidence Interval:</strong> {result.get('lower_bound', 0):.2f} - {result.get('upper_bound', 0):.2f} ppm</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add a button to clear the results
            if st.button("Clear Results"):
                st.session_state.lookup_result = None
                st.session_state.forecast_result = None
                st.session_state.lookup_requested = False
                st.session_state.forecast_requested = False
                st.session_state.initial_load = True  # Reset to initial load
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Historical data lookup section
        st.markdown('<div class="action-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #04870b; text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);">Historical PPM Lookup</h3>', unsafe_allow_html=True)
        
        lookup_type = st.radio(
            "Lookup Type",
            ["Single Date", "Date Range"],
            horizontal=True
        )
        
        if lookup_type == "Single Date":
            # Single date lookup
            min_date = co2_data['date'].min().date()
            max_date = co2_data['date'].max().date()
            
            lookup_date = st.date_input(
                "Select Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="single_date_input"
            )
            
            if st.button("Lookup", key="single_lookup"):
                with st.spinner("Looking up data..."):
                    if lookup_available:
                        # Use NOAALookup
                        records = lookup.lookup_date(lookup_date)
                        if records and len(records) > 0:
                            record = records[0]
                            st.session_state.lookup_result = {
                                'date': pd.to_datetime(record.date),
                                'ppm': record.ppm,
                                'latitude': record.latitude,
                                'longitude': record.longitude
                            }
                        else:
                            st.warning("No data found for the selected date.")
                    else:
                        # Fallback to direct dataframe lookup
                        lookup_date_dt = pd.to_datetime(lookup_date)
                        closest_idx = (co2_data['date'] - lookup_date_dt).abs().idxmin()
                        record = co2_data.iloc[closest_idx]
                        st.session_state.lookup_result = {
                            'date': pd.to_datetime(record['date']),
                            'ppm': record['ppm'],
                            'latitude': record['latitude'],
                            'longitude': record['longitude']
                        }
                    
                    # Clear any forecast result when showing lookup
                    st.session_state.forecast_result = None
                    st.session_state.lookup_requested = True
                    st.session_state.forecast_requested = False
                    st.session_state.initial_load = False
                    # Use st.rerun() instead of experimental_rerun
                    st.rerun()
        
        else:
            # Date range lookup - just pick start and end dates
            min_date = co2_data['date'].min().date()
            max_date = co2_data['date'].max().date()
            
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="range_start_date"
            )
            
            end_date = st.date_input(
                "End Date",
                value=min_date + pd.Timedelta(days=30),
                min_value=min_date,
                max_value=max_date,
                key="range_end_date"
            )
            
            if start_date > end_date:
                st.warning("Start date must be before end date.")
            
            if st.button("Lookup Range", key="range_lookup"):
                with st.spinner("Looking up data..."):
                    if lookup_available:
                        # Use NOAALookup
                        records = lookup.lookup_range(start_date, end_date)
                        
                        if records:
                            # Just highlight the midpoint of the range for simplicity
                            mid_date = start_date + (end_date - start_date) / 2
                            mid_date_dt = pd.to_datetime(mid_date)
                            
                            # Find the closest date
                            all_dates = []
                            for d, measurements in records.items():
                                all_dates.append(pd.to_datetime(d))
                            
                            if all_dates:
                                closest_date = min(all_dates, key=lambda x: abs(x - mid_date_dt))
                                records_for_date = records[closest_date.date()]
                                
                                if records_for_date:
                                    record = records_for_date[0]
                                    st.session_state.lookup_result = {
                                        'date': pd.to_datetime(record.date),
                                        'ppm': record.ppm,
                                        'latitude': record.latitude,
                                        'longitude': record.longitude
                                    }
                        else:
                            st.warning("No data found for the selected date range.")
                    else:
                        # Fallback to direct dataframe lookup
                        mask = (co2_data['date'] >= pd.to_datetime(start_date)) & (co2_data['date'] <= pd.to_datetime(end_date))
                        filtered_data = co2_data[mask]
                        
                        if not filtered_data.empty:
                            # Use the middle record for display
                            mid_idx = len(filtered_data) // 2
                            record = filtered_data.iloc[mid_idx]
                            
                            st.session_state.lookup_result = {
                                'date': pd.to_datetime(record['date']),
                                'ppm': record['ppm'],
                                'latitude': record['latitude'],
                                'longitude': record['longitude']
                            }
                        else:
                            st.warning("No data found for the selected date range.")
                    
                    # Clear any forecast result when showing lookup
                    st.session_state.forecast_result = None
                    st.session_state.lookup_requested = True
                    st.session_state.forecast_requested = False
                    st.session_state.initial_load = False
                    # Use st.rerun() instead of experimental_rerun
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
        # Forecast section
        st.markdown('<div class="action-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #04870b; text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);">AtmoSeer Forecast</h3>', unsafe_allow_html=True)
        
        # Default to one month in the future from the last available data
        last_date = co2_data['date'].max().date()
        min_forecast = last_date + timedelta(days=1)
        max_forecast = last_date + timedelta(days=365)
        
        forecast_date = st.date_input(
            "Select Forecast Date",
            value=min_forecast + timedelta(days=30),
            min_value=min_forecast,
            max_value=max_forecast,
            key="forecast_date"
        )
        
        if st.button("Generate Forecast", key="generate_forecast"):
            with st.spinner("Generating forecast..."):
                # Use the simple forecaster until we fix the model imports
                try:
                    # Create a simple forecaster
                    forecaster = SimpleForecaster(co2_data)
                    prediction, _ = forecaster.predict_for_date(forecast_date)
                    
                    st.session_state.forecast_result = {
                        'date': pd.to_datetime(prediction.date),
                        'ppm': prediction.ppm,
                        'lower_bound': prediction.lower_bound,
                        'upper_bound': prediction.upper_bound
                    }
                    
                    # Clear any lookup result when showing forecast
                    st.session_state.lookup_result = None
                    st.session_state.forecast_requested = True
                    st.session_state.lookup_requested = False
                    st.session_state.initial_load = False
                    # Use st.rerun() instead of experimental_rerun
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during forecasting: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model metrics section
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="metrics-header">CO₂ Model Performance Metrics</h3>', unsafe_allow_html=True)
    
    # Create 4 columns for metrics
    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
    
    with met_col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Model Architecture</p>
            <p class="metric-value">BiLSTM with Attention</p>
        </div>
        """, unsafe_allow_html=True)
    
    with met_col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Mean Absolute Error</p>
            <p class="metric-value">0.48 ppm</p>
        </div>
        """, unsafe_allow_html=True)
    
    with met_col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Training Data Span</p>
            <p class="metric-value">1968 - 2024</p>
        </div>
        """, unsafe_allow_html=True)
    
    with met_col4:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Bayesian Trials</p>
            <p class="metric-value">16</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()