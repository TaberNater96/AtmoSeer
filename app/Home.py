import streamlit as st
import os
import sys
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_base64_of_bin_file(bin_file):
    """
    Convert a binary file to base64 string for embedding in HTML/CSS.
    """
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_image(image_file):
    """
    Set an image as the page background using CSS.
    """
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f'''
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
            background-color: rgba(25, 161, 154, 0.25) !important;
        }}
        
        section[data-testid="stSidebar"] li {{
            padding: 12px 20px;
            margin-bottom: 5px;  /* Add space between buttons */
            font-size: 1.4rem;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 1.0); 
        }}
        
        section[data-testid="stSidebar"] li:hover {{
            border-left: 3px solid #19a19a;
            background-color: rgba(25, 161, 154, 1.0);
            border-radius: 4px;
        }}
        
        .title {{
            font-size: 10rem;
            font-weight: bold;
            text-align: center;
            color: #19a19a;
            margin-bottom: 5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        .title_desc {{
            font-size: 4rem;
            font-weight: bold;
            text-align: center;
            color: #19a19a;
            margin-bottom: 5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
        }}
        
        .header-container {{
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding: 0 1rem;
            width: 100%; 
        }}
        
        .author-name {{
            margin: 0;
            color: #FFFFFF;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
            font-size: 2.5rem;
            font-weight: bold;
        }}
        
        .github-link {{
            display: inline-flex;
            align-items: center;
            background-color: #000000; 
            color: white !important;
            padding: 0.5rem 1rem;
            font-size: 1.5rem;
            text-decoration: none !important;
            border-radius: 5px;
            font-weight: bold;
        }}
        
        .project-description {{
            margin-bottom: 0.1rem;
        }}
        
        .project-description-header {{
            color: #19a19a;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
            font-size: 2.4rem;
            text-align: center;
            font-weight: bold;
            margin-left: 1.0rem;
        }}
        
        .project-description p {{
            background-color: rgba(255, 255, 255, 0.5);
            padding: 1.0rem;
            border-radius: 10px;
            color: #000000;
            font-weight: bold; 
            font-size: 1.4rem;
            line-height: 1.5;
        }}
        
        .data-source-header {{
            color: #19a19a;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
            font-size: 2.4rem;
            text-align: center;
            font-weight: bold;
        }}
        
        .data-source {{
            background-color: rgba(255, 255, 255, 0.5);
            padding: 1.0rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        
        .data-source p {{
            text-align: left;
            line-height: 1.5;
            margin-bottom: 1rem;
            color: #000000;
            font-weight: bold;
            font-size: 1.4rem;
        }}
        
        .data-source a {{
            color: white;
            text-decoration: none;
            font-weight: bold;
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
            background-color: #2980b9; 
            display: inline-block;
            margin-top: auto;
        }}
        
        .data-source a:hover {{
            background-color: #1c5a85; 
        }}

        .logo-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 0.75rem;
            min-height: 120px;
        }}
        
        /* Hide default Streamlit elements */
        #MainMenu, footer, header {{
            visibility: hidden;
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

        .header-anchor-link {{
            display: none !important;
        }}
        
        .pipeline-header {{
            text-align: center;
            color: #19a19a;
            margin-top: 2rem;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
            font-weight: bold;
            font-size: 5.0rem;
        }}
        
        .pipeline-image {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
        }}
        
        .pipeline-text p {{
            background-color: rgba(255, 255, 255, 0.5);
            padding: 0.5rem;
            border-radius: 10px;
            color: #000000;
            font-weight: bold; 
            font-size: 1.35rem;
            line-height: 1.5;
        }}
        
        .analytics-image {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
        }}
        
        .analytics-text p {{
            background-color: rgba(255, 255, 255, 0.5);
            padding: 0.5rem;
            border-radius: 10px;
            color: #000000;
            font-weight: bold; 
            font-size: 1.35rem;
            line-height: 1.5;
        }}
        
        .dl-image {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
        }}
        
        .dl-text p {{
            background-color: rgba(255, 255, 255, 0.5);
            padding: 0.5rem;
            border-radius: 10px;
            color: #000000;
            font-weight: bold; 
            font-size: 1.35rem;
            line-height: 1.5;
        }}
        
        .front-image {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
        }}
        
        .front-text p {{
            background-color: rgba(255, 255, 255, 0.5);
            padding: 0.5rem;
            border-radius: 10px;
            color: #000000;
            font-weight: bold; 
            font-size: 1.35rem;
            line-height: 1.5;
        }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Page Configuration - must be the first Streamlit command
st.set_page_config(
    page_title="AtmoSeer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, "images")
earth_img_path = os.path.join(images_dir, "Earth.jpg")

set_background_image(earth_img_path)

st.sidebar.markdown("""
                    <h1 style='text-align: center; 
                    font-size: 1.4rem;
                    color:#19a19a; 
                    margin-top: 115px;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 1.0);'>Navigate Through Each Gas Type</h1>
                    """, unsafe_allow_html=True)

st.markdown('<h1 class="title">AtmoSeer</h1>', unsafe_allow_html=True)
st.markdown('<h1 class="title_desc">Where Atmospheric Science Meets Deep Learning Algorithms</h1>', unsafe_allow_html=True)

st.markdown(f"""
<div class="header-container">
    <h4 class="author-name">Author: Elijah Taber</h4>
    <a href="https://github.com/TaberNater96/AtmoSeer" target="_blank" class="github-link">
        Source Code &nbsp;<img src="data:image/png;base64,{get_base64_of_bin_file(os.path.join(images_dir, "github_logo.png"))}" width="25">
    </a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="project-description">
    <p>
        AtmoSeer is an extremely comprehensive full-stack data science project that brings predictive atmospheric science to 
        the fingertips of anyone. Unlike most existing tools that only analyze historical greenhouse gas data, AtmoSeer leverages 
        an advanced deep learning algorithm to forecast future emission levels with statistical confidence intervals. 
        <br>
        <br>
        This project represents a complete data pipeline: from extracting raw measurements directly from NOAA's Global Monitoring Laboratory 
        and NASA MODIS databases, through advanced data engineering, to deploying state-of-the-art time series analysis models. 
        AtmoSeer tracks four of some of the most potent greenhouse gases that drive climate change: CO₂, CH₄, N₂O, and SF₆.
        <br>
        <br>
        At the core of AtmoSeer is a custom-built Bidirectional LSTM neural network architecture with an attention mechanism, optimized 
        through Bayesian hyperparameter tuning. This model captures both long-term trends and seasonal patterns in atmospheric gas 
        concentrations through cyclical seasonal awareness that was created during feature engineering, which pairs perfectly with 
        the LSTM's ability to learn both long-term trends going decades back and recent patterns that from only a few week prior. 
        <br>
        <br>
        This project was designed as an open-source contribution to climate science, providing researchers, educators, and concerned citizens 
        with powerful tools to understand and anticipate atmospheric changes that shape Earth's future. 
    </p>
</div>
<h4 class="project-description-header">Forecasting Earth's Atmospheric Future</h4>
<div class="project-description">
    <p>
        To get started, simply select a gas type from the sidebar to get some background on the gas and view some the historical 
        emission trends. To lookup a ppm value for a specific date, select the date from the dropdown menu next to the graph, it has
        the option to choose a single date or a range of dates. The graph will update to show all historical data up to that point as
        well as output the ppm value for that date (or range of dates) underneath the graph. Then to forecast future emission levels
        for a specifc date, use AtmoSeer forecaster (underneath the ppm lookup) to generate a forecast for the selected date. 
        <br>
        <br>
        AtmoSeer will generate predictions for each day past the last recorded date up to the selected date, so the further out the forecast is,
        the longer the model will take to generate the forecast. At the time of this project's completion, there is only data 
        available up to a certain date depending on the gas type, since NOAA does not release real time data in their Global Monitoring Laboratory. Therefore,
        for each gas type, there will be a range of dates that can be looked up, and all dates past that should be forecasted. For example
        CO₂ data is available up to May 31, 2024, so any date past that will need to be generated using AtmoSeer.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
    <div class="logo-container" style="width: 48%;">
        <img src="data:image/png;base64,{}" width="150">
    </div>
    <div class="logo-container" style="width: 48%;">
        <img src="data:image/png;base64,{}" width="250">
    </div>
</div>
""".format(
    get_base64_of_bin_file(os.path.join(images_dir, "noaa_logo.png")),
    get_base64_of_bin_file(os.path.join(images_dir, "nasa_logo.png"))
), unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("""
    <div class="data-source-header">NOAA GML</div>
    <div class="data-source">
        <p>
            Primary source for atmospheric greenhouse gas measurements spanning from back to January of 1968. All datasets
            were gathered from ground based measurement stations, that were spread out across the globe. Not all gases were 
            recorded at the same time, so lookup dates will vary depending on the gas type.
        </p>
        <div style="margin-top: auto; text-align: center;">
            <a href="https://gml.noaa.gov/aftp/data/greenhouse_gases/" target="_blank">Data Repository</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="data-source-header">NASA MODIS</div>
    <div class="data-source">
        <p>
            Satellite data used for biomass density calculations and environmental correlations.
            Gathered and calculated during the feature engineering phase to enhace AtmoSeer by incorporating vegetation 
            patterns into the models. An account with NASA MODIS is required to access the data and access their API.
        </p>
        <div style="margin-top: auto; text-align: center;">
            <a href="https://modis.gsfc.nasa.gov/data/" target="_blank">Data Repository</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
st.markdown("<h2 class='pipeline-header'>Project Pipeline</h2>", unsafe_allow_html=True)

pipeline_col1, pipeline_col2 = st.columns(2, gap="medium")

with pipeline_col1:
    st.markdown("""
    <div class="pipeline-text">
        <p>
            AtmoSeer's foundation began with a robust Extract-Transform-Load pipeline that sourced greenhouse gas measurements 
            directly from NOAA's Global Monitoring Laboratory and vegetation density data from NASA MODIS satellites. The
            extraction process used custom-built web scrapers and API interfaces to gather decades of atmospheric measurements 
            from monitoring stations worldwide. 
            <br>
            <br>
            This raw data was transformed and cleaned using data engineering techniques such
            as imputing missing values using temporal interpolation, standardizing measurements to ensure all data was on the same
            scale such as turning hourly measurements into daily measurements, verifying data quality, and detecting outliers. This 
            data is then loaded into a PostgreSQL database that serves as the central data warehouse, with AWS as a backup cloud database.
        </p>
    </div>
    """, unsafe_allow_html=True)

with pipeline_col2:
    etl_img_path = os.path.join(images_dir, "Data_Engineering.png")
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <img src="data:image/jpeg;base64,{get_base64_of_bin_file(etl_img_path)}" class="pipeline-image">
    </div>
    """, unsafe_allow_html=True)
    
analysis_col1, analysis_col2 = st.columns(2, gap="medium")

with analysis_col1:
    st.markdown("""
    <div class="analytics-text">
        <p>
            The exploratory data analysis phase is used to identify trends, patterns, and correlations in the data. This is
            critical in understanding what is driving the dynamics of greenhouse gas concentrations in the atmosphere. Once insight
            into the data is gained, features such as vegetation density, cyclical encodings, and lagged variations are engineered.
        </p>
    </div>
    """, unsafe_allow_html=True)

with analysis_col2:
    analysis_img_path = os.path.join(images_dir, "Data_Analytics.png")
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <img src="data:image/jpeg;base64,{get_base64_of_bin_file(analysis_img_path)}" class="analytics-image">
    </div>
    """, unsafe_allow_html=True)
    
dl_col1, dl_col2 = st.columns(2, gap="medium")

with dl_col1:
    st.markdown("""
    <div class="dl-text">
        <p>
            As stated before, the foundation of AtmoSeer is a custom-built Bidirectional LSTM neural network architecture with a
            Bayesian Optimization algorithm that acts as a wrapper for hyperparameter tuning. The core architecture of AtmoSeer 
            allows the BiLSTM to capture both long-term trends and seasonal patterns in BOTH directions, allowing it to learn how
            past patterns influence future patterns. Layer normalization, dropout rates, attention mechanisms, and learning rate
            warmups were all used to guide the model towards its most optimal state. 
            <br>
            <br>
            The Bayesian Optimization framework is designed
            to intelligently search for the best set of hyperparameters using Bayes' Theorem to use past results to influence future results.
            It achieves this by taking a hyperparameter space and seeing how the model performs when hyperparameters are increased 
            or decreased, influencing the next set of hyperparameters to be tested. During training GPU acceleration and memory management
            techniques were implemented to greatly increase training speed while reducing memory usage.
        </p>
    </div>
    """, unsafe_allow_html=True)

with dl_col2:
    dl_img_path = os.path.join(images_dir, "DL_Network.png")
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <img src="data:image/jpeg;base64,{get_base64_of_bin_file(dl_img_path)}" class="dl-image">
    </div>
    """, unsafe_allow_html=True)
    
front_col1, front_col2 = st.columns(2, gap="medium")

with front_col1:
    st.markdown("""
    <div class="front-text">
        <p>
            Once all 4 gas types have been gathered and used to train their own specific model, each of the models are woven into 
            streamlit's front end framework to create a friendly user interface that allows users to interact with AtmoSeer to 
            forecast future ppm values as well as look up historical ppm values. Here, users can get background on each of the 
            gas types and see the full history of how each of these gases have been increasing year by year and gain an intuition
            on how these gas concentrations will continue to increase into the future.
        </p>
    </div>
    """, unsafe_allow_html=True)

with front_col2:
    front_img_path = os.path.join(images_dir, "Front_End.png")
    
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <img src="data:image/jpeg;base64,{get_base64_of_bin_file(front_img_path)}" class="front-image">
    </div>
    """, unsafe_allow_html=True)