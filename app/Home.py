import streamlit as st
import os
import sys
import base64

# Add the parent directory to the path to access other modules
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
            flex-direction: row;  /* Changed from column to row */
            justify-content: space-between;  /* Push items to opposite ends */
            align-items: center;
            margin-bottom: 2rem;
            padding: 0 1rem;
            width: 100%;  /* Ensure it spans full width */
        }}
        
        .author-name {{
            margin: 0;
            color: #ba5803;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
            font-size: 2.5rem;
            font-weight: bold;
        }}
        
        .github-link {{
            display: inline-flex;
            align-items: center;
            background-color: #000000;  /* Black background */
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
        
        .project-description p {{
            color: #19a19a;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 1.0);
            font-size: 1.8rem;
            line-height: 1.5;
            font-weight: bold;
            margin-left: 1.0rem;
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
        
        .data-source h3 {{
            text-align: center;
            color: #0A3161;  /* Navy blue */
            font-weight: bold;
        }}
        
        .data-source p {{
            text-align: left;
            line-height: 1.5;
            margin-bottom: 1rem;
            color: #000000;
            font-weight: bold; 
        }}
        
        .data-source a {{
            color: white;
            text-decoration: none;
            font-weight: bold;
            padding: 0.3rem 0.8rem;
            border-radius: 4px;
            background-color: #2980b9;  /* Darker blue */
            display: inline-block;
            margin-top: auto;
        }}
        
        .data-source a:hover {{
            background-color: #1c5a85;  /* Even darker on hover */
        }}

        /* Logo container styling */
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

        /* Alternative approach to hide anchor links */
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
            line-height: 1.8;
        }}
        
        .analytics-image {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
        }}
        
        .analytics-text p {{
            line-height: 1.8;
        }}
        
        .dl-image {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
        }}
        
        .dl-text p {{
            line-height: 1.8;
        }}
        
        .front-image {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.6);
        }}
        
        .front-image p {{
            line-height: 1.8;
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

# Get the absolute path to the images directory
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(current_dir, "images")
earth_img_path = os.path.join(images_dir, "Earth.jpg")

# Set the background image with proper path
set_background_image(earth_img_path)

# Header Section with author and GitHub link in a row
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
        AtmoSeer is a comprehensive environmental data science project focused on greenhouse gas prediction. Using advanced machine learning 
        techniques and data from NOAA's Global Monitoring Laboratory, AtmoSeer analyzes historical greenhouse gas measurements to predict 
        future concentrations of CO2, CH4, N2O, and SF6.
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
    <div class="data-source">
        <h3>NOAA GML</h3>
        <p>
            Primary source for atmospheric greenhouse gas measurements spanning back to 1968.
            Provides global coverage with high precision measurements from monitoring stations worldwide.
        </p>
        <div style="margin-top: auto; text-align: center;">
            <a href="https://gml.noaa.gov/aftp/data/greenhouse_gases/" target="_blank">Data Repository</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="data-source">
        <h3>NASA MODIS</h3>
        <p>
            Satellite data used for biomass density calculations and environmental correlations.
            Enhances predictive capability by incorporating vegetation patterns into models.
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
    <div class="project-description pipeline-text">
        <p>
            The AtmoSeer pipeline follows a systematic ETL (Extract, Transform, Load) approach:
            <br><br>
            <strong>Extract:</strong> Data is sourced from NOAA GML and NASA MODIS cloud databases using web scraping techniques and API requests.
            <br><br>
            <strong>Transform:</strong> The extracted data undergoes several processing steps using Python and Pandas, including handling missing values, detecting duplicates, coercing data types, verifying data quality, and chronological ordering.
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
    <div class="project-description analytics-text">
        <p>
            The AtmoSeer pipeline follows a systematic ETL (Extract, Transform, Load) approach:
            <br><br>
            <strong>Extract:</strong> Data is sourced from NOAA GML and NASA MODIS cloud databases using web scraping techniques and API requests.
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
    <div class="project-description dl-text">
        <p>
            The AtmoSeer pipeline follows a systematic ETL (Extract, Transform, Load) approach:
            <br><br>
            <strong>Extract:</strong> Data is sourced from NOAA GML and NASA MODIS cloud databases using web scraping techniques and API requests.
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
    <div class="project-description front-text">
        <p>
            The AtmoSeer pipeline follows a systematic ETL (Extract, Transform, Load) approach:
            <br><br>
            <strong>Extract:</strong> Data is sourced from NOAA GML and NASA MODIS cloud databases using web scraping techniques and API requests.
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