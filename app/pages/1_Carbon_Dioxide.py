import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import sys
import time
import base64
from datetime import timedelta
import altair as alt
from utils.ppm_lookup import NOAALookup
from atmoseer.atmoseer_core import BayesianTuner
from atmoseer.preprocessors.forecast_setup import CO2_CH4ForecastHelper

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

atmoseer_dir = os.path.join(project_root, 'atmoseer')
sys.path.append(atmoseer_dir)

st.set_page_config(
    page_title="Carbon Dioxide | AtmoSeer",
    page_icon="🌲",
    layout="wide"
)

st.markdown("""
<style>
    .element-container:empty {
        display: none !important;
    }
    
    div:empty {
        display: none !important;
    }
    
    /* Hide Streamlit's default empty containers */
    [data-testid="stAppViewContainer"] div:empty {
        display: none !important;
    }
    
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

current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, "../images")

st.sidebar.markdown("""
                    <h1 style='text-align: center; 
                    font-weight: bold;
                    font-size: 2.4rem;
                    color:#04870b; 
                    margin-top: 5px;
                    margin-bottom:5px;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 1.0);'>CO₂</h1>
                    """, unsafe_allow_html=True)

st.sidebar.markdown(f"""
                    <div style="text-align: center;">
                        <img src="data:image/png;base64,{get_base64_of_bin_file(os.path.join(images_dir, "co2_molecule.png"))}" width="200">
                    </div>
                    """, unsafe_allow_html=True)

st.sidebar.markdown("""
                    <h1 style='text-align: left; 
                    font-size: 1.2rem;
                    color:#FFFFFF; 
                    margin-top:10px;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 1.0);'><strong>Molar Mass:</strong> 44.01 g/mol</h1>
                    """, unsafe_allow_html=True)

st.sidebar.markdown("""
                    <h1 style='text-align: left; 
                    font-size: 1.2rem;
                    color:#FFFFFF; 
                    margin-top: 0.5px;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 1.0);'><strong>Heat Capacity:</strong> 37.11 J/mol·K</h1>
                    """, unsafe_allow_html=True)

st.sidebar.markdown("""
                    <h1 style='text-align: left; 
                    font-size: 1.2rem;
                    color:#FFFFFF; 
                    margin-top: 0.5px;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 1.0);'><strong>Density:</strong> 1.84 kg/m³</h1>
                    """, unsafe_allow_html=True)

st.sidebar.markdown("""
                    <h1 style='text-align: left; 
                    font-size: 1.2rem;
                    color:#FFFFFF; 
                    margin-top: 0.5px;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 1.0);'><strong>Solubility:</strong> 1.45 g/L</h1>
                    """, unsafe_allow_html=True)

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
        
        section[data-testid="stSidebar"] {{
            background-color: rgba(4, 135, 11, 0.25) !important;
        }}
        
        section[data-testid="stSidebar"] li {{
            padding: 8px 20px;
            margin-bottom: 2px;
            font-size: 1.2rem;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 1.0); 
        }}
        
        section[data-testid="stSidebar"] li:hover {{
            border-left: 3px solid #04870b;
            background-color: rgba(4, 135, 11, 1.0);
            border-radius: 4px;
        }}
        
        /* Hide all annoying streamlit header anchor links */
        .css-1629p8f h1 a, .css-1629p8f h2 a, .css-1629p8f h3 a, 
        .css-1629p8f h4 a, .css-1629p8f h5 a, .css-1629p8f h6 a,
        .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a, 
        .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a,
        a.anchor, a.header-anchor {{
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
            position: absolute !important;
        }}
        
        /* Alternative approach to hide anchor links */
        .header-anchor-link {{
            display: none !important;
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
            padding: 1.0rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
            margin-bottom: 1rem;
        }}
        
        .info-container p {{
            font-size: 1.2rem;
            line-height: 1.0;
            color: #FFFFFF;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
            margin-bottom: 1rem;
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
        
        .form-label {{
            font-size: 1.0rem;
            font-weight: bold;
            color: #04870b;
            margin-bottom: 0.01rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        .results-container {{
            background-color: rgba(255, 255, 255, 0.6);
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
            background-color: rgba(255, 255, 255, 0.6);
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
            padding: 0.1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }}
        
        .metric-title {{
            text-align: center;
            font-size: 1.0rem;
            font-weight: bold;
            color: #FFFFFF;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        .metric-value {{
            text-align: center;
            font-size: 1.2rem;
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
        data_paths = [
            os.path.join(project_root, 'data', 'data_warehouse', 'co2_data.csv'),
            os.path.join(project_root, 'data', 'data_warehouse', 'CO2DataNOAA.csv')
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                print(f"Loading data from {path}")
                df = pd.read_csv(path)
                df['date'] = pd.to_datetime(df['date'])
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

@st.cache_resource
def load_co2_model():
    """Load the trained CO2 model with caching."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        models_dir = os.path.join(project_root, 'atmoseer', 'models')
        
        print(f"Looking for model at: {os.path.join(models_dir, 'co2', 'best_model', 'model.pth')}")
        
        if not os.path.exists(os.path.join(models_dir, 'co2', 'best_model', 'model.pth')):
            print(f"Model file does not exist at expected location")
            return None
        
        model = BayesianTuner.load_best_model(
            gas_type='co2', 
            models_dir=models_dir,
            device=device
        )
        return model
    
    except Exception as e:
        st.warning(f"Could not load CO2 model: {str(e)}. Using simple forecaster instead.")
        import traceback
        st.warning(f"Traceback: {traceback.format_exc()}")
        return None

# Define a simple forecast result class for when the actual model can't be loaded
class SimpleForecaster:
    def __init__(self, data):
        self.data = data
    
    def predict_for_date(self, target_date):
        """
        Generate a simple forecast based on historical trends.
        """
        last_date = self.data['date'].max()
        last_ppm = self.data.loc[self.data['date'] == last_date, 'ppm'].values[0]
        
        days_diff = (pd.to_datetime(target_date) - pd.to_datetime(last_date)).days
        
        yearly_increase = 2.5
        daily_increase = yearly_increase / 365
        
        predicted_ppm = last_ppm + (daily_increase * days_diff)
        
        class Prediction:
            def __init__(self, date, ppm, lower, upper):
                self.date = date
                self.ppm = ppm
                self.lower_bound = lower
                self.upper_bound = upper
        
        uncertainty = 1.0 + (days_diff * 0.01)
        prediction = Prediction(
            date=target_date,
            ppm=predicted_ppm,
            lower=predicted_ppm - uncertainty,
            upper=predicted_ppm + uncertainty
        )
        
        return prediction, {}

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
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "../images")
    forest_img_path = os.path.join(images_dir, "Forest.jpg")
    
    if os.path.exists(forest_img_path):
        set_background_image(forest_img_path)
    
    st.markdown('<h1 class="page-title">Carbon Dioxide (CO₂)</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="info-container">
            <p>
                Carbon dioxide (CO₂) is a naturally occurring greenhouse gas composed of one carbon atom and two oxygen atoms. 
                It plays a very important role in Earth's climate system by trapping heat in the atmosphere through the greenhouse effect,
                which helps maintain stable temperatures. However, since the Industrial Revolution in the late 18th century, human
                activities have significantly increased CO₂ levels. The widespread use of fossil fuels, coal, oil, and natural 
                gas—for energy production, transportation, and industry has been the primary driver of this increase. The combustion
                of these fuels releases large amounts of CO₂, which had been stored underground for millions of years, into the
                atmosphere. Deforestation has further exacerbated the issue by reducing the number of trees that can 
                absorb CO₂ through photosynthesis. As a result, atmospheric CO₂ concentrations have risen from about 280 parts per
                million (ppm) before the Industrial Revolution to over 415 ppm today, with a trend that is going nowhere but up.
            </p>
            <p>
                CO₂ accumulates in the atmosphere, it traps more heat, altering climate patterns and threatening ecosystems. An 
                example of the dangers of this runaway greenhouse can be seen in Earth’s nearest neighbor, Venus. Despite being 
                farther from the Sun than Mercury, Venus is the hottest planet in the solar system, with surface temperatures 
                exceeding 900°F (475°C). This extreme heat is due to Venus’ dense atmosphere, composed of over 96% CO₂, which
                creates an intense greenhouse effect. The planet’s atmospheric pressure is approximately 92 times that of Earth, 
                that’s enough pressure to flatten a car like a pancake. The properties of carbon dioxide cause it to remain in the
                atmosphere for thousands of years, meaning its effects on the climate persist long after it is emitted. Even if 
                all CO₂ emissions were to stop this very second, it would take thousands of years for atmospheric levels to return
                to pre-industrial concentrations.
            </p>
        """, unsafe_allow_html=True)
    
    co2_data = load_co2_data()
    
    if co2_data.empty:
        st.error("No data available. Please check the data files in the data_warehouse directory.")
        return
    
    lookup = init_lookup(co2_data)
    lookup_available = lookup is not None
    
    # Main content area - 75% / 25% split
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #04870b; text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);">CO₂ Concentrations (From 1968)</h3>', unsafe_allow_html=True)
        
        chart_container = st.container()
        
        # Only show and animate the chart when specifically requested via buttons
        show_chart = st.session_state.initial_load or st.session_state.lookup_requested or st.session_state.forecast_requested
        
        if show_chart:
            filtered_data = co2_data.copy()
            show_forecast = False
            forecast_point = None
            
            filtered_data = filtered_data.sort_values('date').groupby('date').first().reset_index()

            if st.session_state.lookup_result is not None and st.session_state.lookup_requested:
                end_date = pd.to_datetime(st.session_state.lookup_result['date'])
                filtered_data = filtered_data[filtered_data['date'] <= end_date].copy()
            
            elif st.session_state.forecast_result is not None and st.session_state.forecast_requested:
                show_forecast = True
                forecast_date = pd.to_datetime(st.session_state.forecast_result['date'])
                forecast_point = pd.DataFrame({
                    'date': [forecast_date],
                    'ppm': [st.session_state.forecast_result['ppm']]
                })
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            initial_data = pd.DataFrame({
                'date': [filtered_data.iloc[0]['date']],
                'ppm': [filtered_data.iloc[0]['ppm']]
            })

            chart = alt.Chart(initial_data).mark_line(color='#04870b').encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('ppm:Q', 
                        title='CO₂ Concentration (ppm)', 
                        scale=alt.Scale(domain=[300, filtered_data['ppm'].max() * 1.05]))
            ).properties(
                width='container',
                height=615
            )

            chart_placeholder = chart_container.altair_chart(chart, use_container_width=True)

            total_points = len(filtered_data)
            num_steps = 100 
            points_per_step = max(1, total_points // num_steps)

            animation_data = initial_data.copy()

            for i in range(points_per_step, total_points, points_per_step):
                step_index = min(i, total_points - 1)
                
                new_point = pd.DataFrame({
                    'date': [filtered_data.iloc[step_index]['date']],
                    'ppm': [filtered_data.iloc[step_index]['ppm']]
                })
                
                animation_data = pd.concat([animation_data, new_point])
                
                updated_chart = alt.Chart(animation_data).mark_line(color='#04870b').encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('ppm:Q', 
                            title='CO₂ Concentration (ppm)', 
                            scale=alt.Scale(domain=[300, filtered_data['ppm'].max() * 1.05]))
                ).properties(
                    width='container',
                    height=615
                )
                chart_placeholder.altair_chart(updated_chart, use_container_width=True)
                
                progress = int((i / total_points) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Loading data: {progress}% complete")
                
                time.sleep(0.01)

            if total_points > 0 and step_index < total_points - 1:
                final_point = pd.DataFrame({
                    'date': [filtered_data.iloc[-1]['date']],
                    'ppm': [filtered_data.iloc[-1]['ppm']]
                })
                animation_data = pd.concat([animation_data, final_point])

            if show_forecast:
                latest_historical_date = filtered_data['date'].max()
                
                last_historical_ppm = filtered_data.loc[filtered_data['date'] == latest_historical_date, 'ppm'].values[0]
                forecast_ppm = st.session_state.forecast_result['ppm']
                
                if 'historical' in st.session_state.forecast_result:
                    historical_forecasts = st.session_state.forecast_result['historical']
                    
                    forecast_data = []
                    for date_str, values in historical_forecasts.items():
                        forecast_data.append({
                            'date': pd.to_datetime(date_str),
                            'ppm': values['ppm'],
                            'lower_bound': values['lower_bound'],
                            'upper_bound': values['upper_bound']
                        })
                    
                    forecast_df = pd.DataFrame(forecast_data)
                    forecast_df = forecast_df.sort_values('date')
                else:
                    date_range = pd.date_range(start=latest_historical_date, end=forecast_date, freq='D')
                    ppm_values = np.linspace(last_historical_ppm, forecast_ppm, len(date_range))
                    
                    forecast_df = pd.DataFrame({
                        'date': date_range,
                        'ppm': ppm_values
                    })
                    
                    lower_bound = st.session_state.forecast_result.get('lower_bound', 0)
                    upper_bound = st.session_state.forecast_result.get('upper_bound', 0)
                    
                    interval_width = (upper_bound - lower_bound) / forecast_ppm
                    
                    forecast_df['lower_bound'] = forecast_df['ppm'] * (1 - interval_width * np.linspace(0, 1, len(forecast_df)))
                    forecast_df['upper_bound'] = forecast_df['ppm'] * (1 + interval_width * np.linspace(0, 1, len(forecast_df)))
                
                forecast_line_layer = alt.Chart(forecast_df).mark_line(
                    color='red',
                    strokeWidth=2
                ).encode(
                    x='date:T',
                    y='ppm:Q'
                )
                
                confidence_interval = alt.Chart(forecast_df).mark_area(
                    color='gray',
                    opacity=0.6
                ).encode(
                    x='date:T',
                    y='lower_bound:Q',
                    y2='upper_bound:Q'
                )
                
                forecast_text = alt.Chart(pd.DataFrame({
                    'date':[forecast_date - pd.Timedelta(days=30)],
                    'ppm': [forecast_ppm],
                    'text': ['Forecast']
                })).mark_text(
                    align='left',
                    baseline='middle',
                    dx=5,
                    fontSize=14,
                    fontWeight='bold',
                    color='red'
                ).encode(
                    x='date:T',
                    y='ppm:Q',
                    text='text:N'
                )
                
                final_chart = alt.Chart(animation_data).mark_line(color='#04870b').encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('ppm:Q', 
                            title='CO₂ Concentration (ppm)', 
                            scale=alt.Scale(domain=[300, max(filtered_data['ppm'].max(), forecast_point['ppm'].max()) * 1.05]))
                ).properties(
                    width='container',
                    height=615
                )
                
                combined_chart = final_chart + confidence_interval + forecast_line_layer + forecast_text
                chart_placeholder.altair_chart(combined_chart, use_container_width=True)
            
            st.session_state.lookup_requested = False
            st.session_state.forecast_requested = False
            st.session_state.initial_load = False
            
            progress_bar.progress(100)
            status_text.text("Data loading complete!")
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()
        else:
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
                height=615
            )
            
            chart_container.altair_chart(placeholder_chart, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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
            
            if st.button("Clear Results"):
                st.session_state.lookup_result = None
                st.session_state.forecast_result = None
                st.session_state.lookup_requested = False
                st.session_state.forecast_requested = False
                st.session_state.initial_load = True 
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="action-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #04870b; text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);">Historical PPM Lookup</h3>', unsafe_allow_html=True)
        st.markdown('<p class="form-label">Lookup Type</p>', unsafe_allow_html=True)
        
        lookup_type = st.radio(
            "",
            ["Single Date", "Date Range"],
            horizontal=True
        )
        
        if lookup_type == "Single Date":
            min_date = co2_data['date'].min().date()
            max_date = co2_data['date'].max().date()
            
            st.markdown('<p class="form-label">Select Date From Jan 1968 to May 2024</p>', unsafe_allow_html=True)
            lookup_date = st.date_input(
                "",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="single_date_input"
            )
            
            if st.button("Lookup", key="single_lookup"):
                with st.spinner("Looking up data..."):
                    if lookup_available:
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
                        lookup_date_dt = pd.to_datetime(lookup_date)
                        closest_idx = (co2_data['date'] - lookup_date_dt).abs().idxmin()
                        record = co2_data.iloc[closest_idx]
                        st.session_state.lookup_result = {
                            'date': pd.to_datetime(record['date']),
                            'ppm': record['ppm'],
                            'latitude': record['latitude'],
                            'longitude': record['longitude']
                        }
                    
                    st.session_state.forecast_result = None
                    st.session_state.lookup_requested = True
                    st.session_state.forecast_requested = False
                    st.session_state.initial_load = False
                    st.rerun()
        
        else:
            min_date = co2_data['date'].min().date()
            max_date = co2_data['date'].max().date()
            
            st.markdown('<p class="form-label">Start Date</p>', unsafe_allow_html=True)
            start_date = st.date_input(
                "",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="range_start_date"
            )
            
            st.markdown('<p class="form-label">End Date</p>', unsafe_allow_html=True)
            end_date = st.date_input(
                "",
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
                        records = lookup.lookup_range(start_date, end_date)
                        
                        if records:
                            mid_date = start_date + (end_date - start_date) / 2
                            mid_date_dt = pd.to_datetime(mid_date)
                            
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
                        mask = (co2_data['date'] >= pd.to_datetime(start_date)) & (co2_data['date'] <= pd.to_datetime(end_date))
                        filtered_data = co2_data[mask]
                        
                        if not filtered_data.empty:
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
                    
                    st.session_state.forecast_result = None
                    st.session_state.lookup_requested = True
                    st.session_state.forecast_requested = False
                    st.session_state.initial_load = False
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
        st.markdown('<div class="action-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #04870b; text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);">AtmoSeer Forecast</h3>', unsafe_allow_html=True)
        
        last_date = co2_data['date'].max().date()
        min_forecast = last_date + timedelta(days=1)
        max_forecast = last_date + timedelta(days=365)
        
        st.markdown('<p class="form-label">Select Date From June 2024 to May 2025</p>', unsafe_allow_html=True)
        forecast_date = st.date_input(
            "",
            value=min_forecast + timedelta(days=30),
            min_value=min_forecast,
            max_value=max_forecast,
            key="forecast_date"
        )
        
        if st.button("Generate Forecast", key="generate_forecast"):
            with st.spinner("Generating forecast..."):
                try:
                    model = load_co2_model()
                    
                    if model is not None:
                        forecaster = CO2_CH4ForecastHelper(model, co2_data)
                        prediction, historical = forecaster.predict_for_date(forecast_date)
                        
                        forecast_values = {
                            date_val.strftime('%Y-%m-%d'): {
                                'ppm': result.ppm,
                                'lower_bound': result.lower_bound, 
                                'upper_bound': result.upper_bound
                            } for date_val, result in historical.items()
                        }
                        
                        st.session_state.forecast_result = {
                            'date': pd.to_datetime(prediction.date),
                            'ppm': prediction.ppm,
                            'lower_bound': prediction.lower_bound,
                            'upper_bound': prediction.upper_bound,
                            'historical': forecast_values 
                        }
                    else:
                        forecaster = SimpleForecaster(co2_data)
                        prediction, _ = forecaster.predict_for_date(forecast_date)
                        
                        st.session_state.forecast_result = {
                            'date': pd.to_datetime(prediction.date),
                            'ppm': prediction.ppm,
                            'lower_bound': prediction.lower_bound,
                            'upper_bound': prediction.upper_bound
                        }
                    
                    st.session_state.lookup_result = None
                    st.session_state.forecast_requested = True
                    st.session_state.lookup_requested = False
                    st.session_state.initial_load = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during forecasting: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc()) 
        
        st.markdown('<p class="form-label">Note that the farther out the forecast date is from the last recorded ppm value (May 31, 2024), the longer it will take AtmoSeer to generate a forecast.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="metrics-header">AtmoSeer CO₂ Model Metrics</h3>', unsafe_allow_html=True)
    
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
            <p class="metric-title">Best Validation Loss</p>
            <p class="metric-value">0.0632</p>
        </div>
        """, unsafe_allow_html=True)
    
    with met_col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">NOAA CO₂ PPM Data Span</p>
            <p class="metric-value">Jan 16, 1968 - May 31, 2024</p>
        </div>
        """, unsafe_allow_html=True)
    
    with met_col4:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Tuner</p>
            <p class="metric-value">Bayesian Optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
    config_col1, config_col2, config_col3, config_col4 = st.columns(4)
    
    with config_col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Hidden Dimensions</p>
            <p class="metric-value">264</p>
        </div>
        """, unsafe_allow_html=True)
        
    with config_col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Number of Layers</p>
            <p class="metric-value">2</p>
        </div>
        """, unsafe_allow_html=True)
        
    with config_col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Learning Rate</p>
            <p class="metric-value">0.0001</p>
        </div>
        """, unsafe_allow_html=True)
        
    with config_col4:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Sequence Length</p>
            <p class="metric-value">30 Days</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()