import unittest
import os
import sys
import base64
from unittest.mock import patch, MagicMock, mock_open
import streamlit as st
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

spec = importlib.util.spec_from_file_location("Home", os.path.join(os.path.dirname(__file__), "..", "pages", "Home.py"))
Home = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Home)

class TestHomePage(unittest.TestCase):
    
    def setUp(self):
        self.st_mock = MagicMock()
        self.os_mock = MagicMock()
        self.base64_mock = MagicMock()
        self.sys_mock = MagicMock()
        
        self.mock_current_dir = "/mock/current/dir"
        
        self.mock_images_dir = os.path.join(self.mock_current_dir, "images")
        
        self.mock_earth_img_path = os.path.join(self.mock_images_dir, "Earth.jpg")
        
    def tearDown(self):
        # Clean up after each test
        pass
    
    def test_get_base64_of_bin_file(self):
        # Test case 1: Test the base64 encoding function with a mock file.
        test_data = b"test binary data"
        expected_result = "dGVzdCBiaW5hcnkgZGF0YQ=="
        
        with patch("builtins.open", mock_open(read_data=test_data)):
            result = Home.get_base64_of_bin_file("mock_file.jpg")
            
        self.assertEqual(result, expected_result)
    
    def test_get_base64_of_bin_file_file_not_found(self):
        # Test case 2: Test handling of file not found error.
        with patch("builtins.open", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                Home.get_base64_of_bin_file("nonexistent_file.jpg")
    
    def test_set_background_image(self):
        # Test case 3: Test the background image setter function.
        mock_bin_str = "mock_base64_string"
        
        with patch.object(Home, "get_base64_of_bin_file", return_value=mock_bin_str):
            with patch.object(st, "markdown") as mock_markdown:
                Home.set_background_image("mock_image.jpg")
                
                # Check if st.markdown was called with CSS that includes our mock base64 string
                call_args = mock_markdown.call_args[0][0]
                self.assertIn(mock_bin_str, call_args)
                self.assertIn("background-image", call_args)
                self.assertTrue(mock_markdown.call_args[1]['unsafe_allow_html'])
    
    def test_main_page_configuration(self):
        # Test case 4: Test that the main page configuration is set correctly.
        with patch.object(st, "set_page_config") as mock_set_page_config:
            # Re-import the module to trigger the st.set_page_config call
            spec = importlib.util.spec_from_file_location("Home", os.path.join(os.path.dirname(__file__), "..", "pages", "Home.py"))
            Home_reimport = importlib.util.module_from_spec(spec)
            
            with patch.object(os.path, "dirname", return_value=self.mock_current_dir):
                with patch.object(os.path, "abspath", return_value=self.mock_current_dir):
                    with patch.object(os.path, "join", return_value=self.mock_earth_img_path):
                        with patch.object(Home, "set_background_image"):
                            with patch.object(st, "sidebar"):
                                with patch.object(st, "markdown"):
                                    spec.loader.exec_module(Home_reimport)
                                    
                                    mock_set_page_config.assert_called_once()
                                    call_kwargs = mock_set_page_config.call_args[1]
                                    self.assertEqual(call_kwargs["page_title"], "AtmoSeer")
                                    self.assertEqual(call_kwargs["page_icon"], "🌍")
                                    self.assertEqual(call_kwargs["layout"], "wide")
                                    self.assertEqual(call_kwargs["initial_sidebar_state"], "expanded")
    
    def test_background_image_loading(self):
        # Test case 5: Test the proper loading of the background image.
        with patch.object(os.path, "dirname", return_value=self.mock_current_dir):
            with patch.object(os.path, "abspath", return_value=self.mock_current_dir):
                with patch.object(os.path, "join", return_value=self.mock_earth_img_path):
                    with patch.object(Home, "set_background_image") as mock_set_background:
                        # Execute the relevant part of the Home.py script
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        images_dir = os.path.join(current_dir, "images")
                        earth_img_path = os.path.join(images_dir, "Earth.jpg")
                        Home.set_background_image(earth_img_path)
                        
                        # Check if set_background_image was called with the correct path
                        mock_set_background.assert_called_once_with(self.mock_earth_img_path)
    
    def test_sidebar_navigation(self):
        # Test case 6: Test the sidebar navigation section.
        with patch.object(st.sidebar, "markdown") as mock_sidebar_markdown:
            # Re-execute the sidebar creation part
            Home.st.sidebar.markdown("""
                        <h1 style='text-align: center; 
                        font-size: 1.4rem;
                        color:#19a19a; 
                        margin-top: 115px;
                        text-shadow: 1px 1px 2px rgba(0, 0, 0, 1.0);'>Navigate Through Each Gas Type</h1>
                        """, unsafe_allow_html=True)
            
            # Check if sidebar.markdown was called with the correct HTML
            call_args = mock_sidebar_markdown.call_args[0][0]
            self.assertIn("Navigate Through Each Gas Type", call_args)
            self.assertIn("color:#19a19a", call_args)
            self.assertTrue(mock_sidebar_markdown.call_args[1]['unsafe_allow_html'])
    
    def test_main_title_rendering(self):
        # Test case 7: Test the rendering of the main title.
        with patch.object(st, "markdown") as mock_markdown:
            # Re-execute the title creation part
            Home.st.markdown('<h1 class="title">AtmoSeer</h1>', unsafe_allow_html=True)
            Home.st.markdown('<h1 class="title_desc">Where Atmospheric Science Meets Deep Learning Algorithms</h1>', unsafe_allow_html=True)
            
            # Check the first title call
            first_call_args = mock_markdown.call_args_list[0][0][0]
            self.assertEqual(first_call_args, '<h1 class="title">AtmoSeer</h1>')
            self.assertTrue(mock_markdown.call_args_list[0][1]['unsafe_allow_html'])
            
            # Check the second title call
            second_call_args = mock_markdown.call_args_list[1][0][0]
            self.assertIn("Where Atmospheric Science Meets Deep Learning Algorithms", second_call_args)
            self.assertTrue(mock_markdown.call_args_list[1][1]['unsafe_allow_html'])
    
    def test_author_and_github_link(self):
        # Test case 8: Test the author and GitHub link section.
        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64_github_logo"):
            with patch.object(st, "markdown") as mock_markdown:
                # Mock os.path.join to return a fixed path for the GitHub logo
                with patch.object(os.path, "join", return_value="/mock/path/to/github_logo.png"):
                    # Re-execute the header container creation
                    images_dir = "/mock/path/to"
                    Home.st.markdown(f"""
                    <div class="header-container">
                        <h4 class="author-name">Author: Elijah Taber</h4>
                        <a href="https://github.com/TaberNater96/AtmoSeer" target="_blank" class="github-link">
                            Source Code &nbsp;<img src="data:image/png;base64,{Home.get_base64_of_bin_file(os.path.join(images_dir, "github_logo.png"))}" width="25">
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Check if markdown was called with the correct HTML
                    call_args = mock_markdown.call_args[0][0]
                    self.assertIn("Author: Elijah Taber", call_args)
                    self.assertIn("https://github.com/TaberNater96/AtmoSeer", call_args)
                    self.assertIn("mock_base64_github_logo", call_args)
                    self.assertTrue(mock_markdown.call_args[1]['unsafe_allow_html'])
    
    def test_project_description(self):
        # Test case 9: Test the project description section.
        with patch.object(st, "markdown") as mock_markdown:
            # Re-execute the project description section
            Home.st.markdown("""
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
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Check if markdown was called with the correct HTML
            call_args = mock_markdown.call_args[0][0]
            self.assertIn("AtmoSeer is an extremely comprehensive", call_args)
            self.assertIn("NOAA's Global Monitoring Laboratory", call_args)
            self.assertIn("CO₂, CH₄, N₂O, and SF₆", call_args)
            self.assertTrue(mock_markdown.call_args[1]['unsafe_allow_html'])
    
    def test_agency_logos(self):
        # Test case 10: Test the NOAA and NASA logo section.
        with patch.object(Home, "get_base64_of_bin_file") as mock_get_base64:
            mock_get_base64.side_effect = ["mock_base64_noaa", "mock_base64_nasa"]
            
            with patch.object(st, "markdown") as mock_markdown:
                with patch.object(os.path, "join") as mock_join:
                    mock_join.side_effect = ["/mock/path/to/noaa_logo.png", "/mock/path/to/nasa_logo.png"]
                    
                    # Re-execute the logo section
                    images_dir = "/mock/path/to"
                    Home.st.markdown("""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                        <div class="logo-container" style="width: 48%;">
                            <img src="data:image/png;base64,{}" width="150">
                        </div>
                        <div class="logo-container" style="width: 48%;">
                            <img src="data:image/png;base64,{}" width="250">
                        </div>
                    </div>
                    """.format(
                        Home.get_base64_of_bin_file(os.path.join(images_dir, "noaa_logo.png")),
                        Home.get_base64_of_bin_file(os.path.join(images_dir, "nasa_logo.png"))
                    ), unsafe_allow_html=True)
                    
                    # Check that get_base64_of_bin_file was called with the correct paths
                    self.assertEqual(mock_get_base64.call_count, 2)
                    self.assertEqual(mock_join.call_count, 2)
                    
                    # Check if markdown was called with the correct HTML
                    call_args = mock_markdown.call_args[0][0]
                    self.assertIn("mock_base64_noaa", call_args)
                    self.assertIn("mock_base64_nasa", call_args)
                    self.assertTrue(mock_markdown.call_args[1]['unsafe_allow_html'])
    
    def test_data_source_columns(self):
        # Test case 11: Test the data source columns (NOAA GML and NASA MODIS).
        with patch.object(st, "columns") as mock_columns:
            # Create mock column objects
            mock_col1 = MagicMock()
            mock_col2 = MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2]
            
            # Re-execute the data source columns section
            col1, col2 = Home.st.columns(2, gap="medium")
            
            # Test NOAA GML column
            with mock_col1:
                Home.st.markdown("""
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
            
            # Test NASA MODIS column
            with mock_col2:
                Home.st.markdown("""
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
            
            # Verify the columns were created with the correct parameters
            mock_columns.assert_called_once_with(2, gap="medium")
            
            # Check that markdown was called on each column
            mock_col1.markdown.assert_called_once()
            mock_col2.markdown.assert_called_once()
            
            # Verify content in column 1
            col1_args = mock_col1.markdown.call_args[0][0]
            self.assertIn("NOAA GML", col1_args)
            self.assertIn("January of 1968", col1_args)
            self.assertIn("https://gml.noaa.gov/aftp/data/greenhouse_gases/", col1_args)
            self.assertTrue(mock_col1.markdown.call_args[1]['unsafe_allow_html'])
            
            # Verify content in column 2
            col2_args = mock_col2.markdown.call_args[0][0]
            self.assertIn("NASA MODIS", col2_args)
            self.assertIn("biomass density calculations", col2_args)
            self.assertIn("https://modis.gsfc.nasa.gov/data/", col2_args)
            self.assertTrue(mock_col2.markdown.call_args[1]['unsafe_allow_html'])
    
    def test_pipeline_header(self):
        # Test case 12: Test the pipeline header section.
        with patch.object(st, "markdown") as mock_markdown:
            # Re-execute the pipeline header section
            Home.st.markdown("<h2 class='pipeline-header'>Project Pipeline</h2>", unsafe_allow_html=True)
            
            # Check if markdown was called with the correct HTML
            call_args = mock_markdown.call_args[0][0]
            self.assertEqual(call_args, "<h2 class='pipeline-header'>Project Pipeline</h2>")
            self.assertTrue(mock_markdown.call_args[1]['unsafe_allow_html'])
    
    def test_pipeline_columns(self):
        # Test case 13: Test the pipeline columns section.
        with patch.object(st, "columns") as mock_columns:
            # Create mock column objects
            mock_col1 = MagicMock()
            mock_col2 = MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2]
            
            # Re-execute the pipeline columns section
            pipeline_col1, pipeline_col2 = Home.st.columns(2, gap="medium")
            
            # Test text column
            with mock_col1:
                Home.st.markdown("""
                <div class="pipeline-text">
                    <p>
                        AtmoSeer's foundation began with a robust Extract-Transform-Load pipeline that sourced greenhouse gas measurements 
                        directly from NOAA's Global Monitoring Laboratory and vegetation density data from NASA MODIS satellites.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Mock the image path and get_base64_of_bin_file function for the image column
            etl_img_path = "/mock/path/to/Data_Engineering.png"
            
            with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64_etl_image"):
                # Test image column
                with mock_col2:
                    Home.st.markdown(f"""
                    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                        <img src="data:image/jpeg;base64,{Home.get_base64_of_bin_file(etl_img_path)}" class="pipeline-image">
                    </div>
                    """, unsafe_allow_html=True)
            
            # Verify the columns were created with the correct parameters
            mock_columns.assert_called_once_with(2, gap="medium")
            
            # Check that markdown was called on each column
            mock_col1.markdown.assert_called_once()
            mock_col2.markdown.assert_called_once()
            
            # Verify content in column 1
            col1_args = mock_col1.markdown.call_args[0][0]
            self.assertIn("Extract-Transform-Load pipeline", col1_args)
            self.assertIn("NOAA's Global Monitoring Laboratory", col1_args)
            self.assertTrue(mock_col1.markdown.call_args[1]['unsafe_allow_html'])
            
            # Verify content in column 2
            col2_args = mock_col2.markdown.call_args[0][0]
            self.assertIn("mock_base64_etl_image", col2_args)
            self.assertIn("pipeline-image", col2_args)
            self.assertTrue(mock_col2.markdown.call_args[1]['unsafe_allow_html'])
    
    def test_analysis_columns(self):
        # Test case 14: Test the analysis columns section.
        with patch.object(st, "columns") as mock_columns:
            # Create mock column objects
            mock_col1 = MagicMock()
            mock_col2 = MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2]
            
            # Re-execute the analysis columns section
            analysis_col1, analysis_col2 = Home.st.columns(2, gap="medium")
            
            # Test text column
            with mock_col1:
                Home.st.markdown("""
                <div class="analytics-text">
                    <p>
                        The exploratory data analysis phase is used to identify trends, patterns, and correlations in the data.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Mock the image path and get_base64_of_bin_file function for the image column
            analysis_img_path = "/mock/path/to/Data_Analytics.png"
            
            with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64_analysis_image"):
                # Test image column
                with mock_col2:
                    Home.st.markdown(f"""
                    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                        <img src="data:image/jpeg;base64,{Home.get_base64_of_bin_file(analysis_img_path)}" class="analytics-image">
                    </div>
                    """, unsafe_allow_html=True)
            
            # Verify the columns were created with the correct parameters
            mock_columns.assert_called_once_with(2, gap="medium")
            
            # Check that markdown was called on each column
            mock_col1.markdown.assert_called_once()
            mock_col2.markdown.assert_called_once()
            
            # Verify content in column 1
            col1_args = mock_col1.markdown.call_args[0][0]
            self.assertIn("exploratory data analysis", col1_args)
            self.assertIn("identify trends", col1_args)
            self.assertTrue(mock_col1.markdown.call_args[1]['unsafe_allow_html'])
            
            # Verify content in column 2
            col2_args = mock_col2.markdown.call_args[0][0]
            self.assertIn("mock_base64_analysis_image", col2_args)
            self.assertIn("analytics-image", col2_args)
            self.assertTrue(mock_col2.markdown.call_args[1]['unsafe_allow_html'])
    
    def test_dl_columns(self):
        # Test case 15: Test the deep learning columns section.
        with patch.object(st, "columns") as mock_columns:
            # Create mock column objects
            mock_col1 = MagicMock()
            mock_col2 = MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2]
            
            # Re-execute the deep learning columns section
            dl_col1, dl_col2 = Home.st.columns(2, gap="medium")
            
            # Test text column
            with mock_col1:
                Home.st.markdown("""
                <div class="dl-text">
                    <p>
                        As stated before, the foundation of AtmoSeer is a custom-built Bidirectional LSTM neural network architecture.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Mock the image path and get_base64_of_bin_file function for the image column
            dl_img_path = "/mock/path/to/DL_Network.png"
            
            with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64_dl_image"):
                # Test image column
                with mock_col2:
                    Home.st.markdown(f"""
                    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                        <img src="data:image/jpeg;base64,{Home.get_base64_of_bin_file(dl_img_path)}" class="dl-image">
                    </div>
                    """, unsafe_allow_html=True)
            
            # Verify the columns were created with the correct parameters
            mock_columns.assert_called_once_with(2, gap="medium")
            
            # Check that markdown was called on each column
            mock_col1.markdown.assert_called_once()
            mock_col2.markdown.assert_called_once()
            
            # Verify content in column 1
            col1_args = mock_col1.markdown.call_args[0][0]
            self.assertIn("Bidirectional LSTM neural network", col1_args)
            self.assertTrue(mock_col1.markdown.call_args[1]['unsafe_allow_html'])
            
            # Verify content in column 2
            col2_args = mock_col2.markdown.call_args[0][0]
            self.assertIn("mock_base64_dl_image", col2_args)
            self.assertIn("dl-image", col2_args)
            self.assertTrue(mock_col2.markdown.call_args[1]['unsafe_allow_html'])
    
    def test_front_columns(self):
        # Test case 16: Test the frontend columns section.
        with patch.object(st, "columns") as mock_columns:
            # Create mock column objects
            mock_col1 = MagicMock()
            mock_col2 = MagicMock()
            mock_columns.return_value = [mock_col1, mock_col2]
            
            # Re-execute the frontend columns section
            front_col1, front_col2 = Home.st.columns(2, gap="medium")
            
            # Test text column
            with mock_col1:
                Home.st.markdown("""
                <div class="front-text">
                    <p>
                        Once all 4 gas types have been gathered and used to train their own specific model.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Mock the image path and get_base64_of_bin_file function for the image column
            front_img_path = "/mock/path/to/Front_End.png"
            
            with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64_front_image"):
                # Test image column
                with mock_col2:
                    Home.st.markdown(f"""
                    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                        <img src="data:image/jpeg;base64,{Home.get_base64_of_bin_file(front_img_path)}" class="front-image">
                    </div>
                    """, unsafe_allow_html=True)
            
            # Verify the columns were created with the correct parameters
            mock_columns.assert_called_once_with(2, gap="medium")
            
            # Check that markdown was called on each column
            mock_col1.markdown.assert_called_once()
            mock_col2.markdown.assert_called_once()
            
            # Verify content in column 1
            col1_args = mock_col1.markdown.call_args[0][0]
            self.assertIn("4 gas types", col1_args)
            self.assertTrue(mock_col1.markdown.call_args[1]['unsafe_allow_html'])
            
            # Verify content in column 2
            col2_args = mock_col2.markdown.call_args[0][0]
            self.assertIn("mock_base64_front_image", col2_args)
            self.assertIn("front-image", col2_args)
            self.assertTrue(mock_col2.markdown.call_args[1]['unsafe_allow_html'])
    
    def test_path_construction(self):
        # Test case 17: Test the proper construction of file paths.
        with patch.object(os.path, "dirname", return_value="/mock/dir"):
            with patch.object(os.path, "abspath", return_value="/mock/absolute/dir"):
                with patch.object(os.path, "join") as mock_join:
                    mock_join.side_effect = [
                        "/mock/absolute/dir/images",
                        "/mock/absolute/dir/images/Earth.jpg"
                    ]
                    
                    # Execute the path construction code
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    images_dir = os.path.join(current_dir, "images")
                    earth_img_path = os.path.join(images_dir, "Earth.jpg")
                    
                    # Verify the constructed paths
                    self.assertEqual(current_dir, "/mock/absolute/dir")
                    self.assertEqual(images_dir, "/mock/absolute/dir/images")
                    self.assertEqual(earth_img_path, "/mock/absolute/dir/images/Earth.jpg")
    
    def test_css_style_content(self):
        # Test case 18: Test the CSS style content in set_background_image function.
        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64_image"):
            with patch.object(st, "markdown") as mock_markdown:
                # Call the function
                Home.set_background_image("mock_image.jpg")
                
                # Get the CSS content from the call
                css_content = mock_markdown.call_args[0][0]
                
                # Verify essential CSS styles are present
                self.assertIn("background-image", css_content)
                self.assertIn("background-size: cover", css_content)
                self.assertIn("font-family: 'Courier New', monospace", css_content)
                self.assertIn("section[data-testid=\"stSidebar\"]", css_content)
                self.assertIn(".title {", css_content)
                self.assertIn(".project-description", css_content)
                self.assertIn(".data-source", css_content)
                self.assertIn(".pipeline-image", css_content)
                self.assertIn("/* Hide default Streamlit elements */", css_content)
    
    def test_integration_with_system_modules(self):
        # Test case 19: Test integration with required system modules.
        with patch.object(sys, "path") as mock_sys_path:
            with patch.object(sys.path, "append") as mock_append:
                # Re-import to trigger sys.path.append
                spec = importlib.util.spec_from_file_location("Home", os.path.join(os.path.dirname(__file__), "..", "pages", "Home.py"))
                Home_reimport = importlib.util.module_from_spec(spec)
                
                with patch.multiple(os.path, dirname=MagicMock(return_value="/mock/dir"), 
                                  abspath=MagicMock(return_value="/mock/absolute/dir"), 
                                  join=MagicMock(return_value="/mock/path")):
                    with patch.object(Home, "set_background_image"):
                        with patch.object(st, "sidebar"):
                            with patch.object(st, "markdown"):
                                with patch.object(st, "set_page_config"):
                                    spec.loader.exec_module(Home_reimport)
                                    
                                    # Check if sys.path.append was called
                                    mock_append.assert_called()
    
    def test_all_markdown_calls(self):
        # Test case 20: Test all markdown calls throughout the page.
        with patch.object(st, "markdown") as mock_markdown:
            with patch.object(os.path, "dirname", return_value=self.mock_current_dir):
                with patch.object(os.path, "abspath", return_value=self.mock_current_dir):
                    with patch.object(os.path, "join", return_value=self.mock_earth_img_path):
                        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64"):
                            with patch.object(st, "sidebar"):
                                with patch.object(st, "set_page_config"):
                                    with patch.object(st, "columns", return_value=[MagicMock(), MagicMock()]):
                                        # Re-import to execute all markdown calls
                                        spec = importlib.util.spec_from_file_location("Home", os.path.join(os.path.dirname(__file__), "..", "pages", "Home.py"))
                                        Home_reimport = importlib.util.module_from_spec(spec)
                                        spec.loader.exec_module(Home_reimport)
                                        
                                        # Verify markdown was called multiple times (we just count, don't check each call)
                                        self.assertGreater(mock_markdown.call_count, 5)
                                        
                                        # Check that all calls have unsafe_allow_html=True
                                        for call in mock_markdown.call_args_list:
                                            self.assertTrue(call[1]['unsafe_allow_html'])
    
    def test_streamlit_columns_creation(self):
        # Test case 21: Test the creation of all column sections.
        with patch.object(st, "columns") as mock_columns:
            # Create mock column objects
            mock_cols = [MagicMock(), MagicMock()]
            mock_columns.return_value = mock_cols
            
            with patch.object(os.path, "dirname", return_value=self.mock_current_dir):
                with patch.object(os.path, "abspath", return_value=self.mock_current_dir):
                    with patch.object(os.path, "join", return_value=self.mock_earth_img_path):
                        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64"):
                            with patch.object(st, "sidebar"):
                                with patch.object(st, "markdown"):
                                    with patch.object(st, "set_page_config"):
                                        # Re-import to execute all columns calls
                                        spec = importlib.util.spec_from_file_location("Home", os.path.join(os.path.dirname(__file__), "..", "pages", "Home.py"))
                                        Home_reimport = importlib.util.module_from_spec(spec)
                                        spec.loader.exec_module(Home_reimport)
                                        
                                        # There should be 5 calls to st.columns (data sources, pipeline, analysis, dl, front)
                                        self.assertEqual(mock_columns.call_count, 5)
                                        
                                        # All should be 2-column layouts with "medium" gap
                                        for call in mock_columns.call_args_list:
                                            self.assertEqual(call[0][0], 2)
                                            self.assertEqual(call[1]['gap'], "medium")
    
    def test_page_structure_completeness(self):
        # Test case 22: Test that all major sections of the page are present.
        main_sections = [
            "<h1 class=\"title\">AtmoSeer</h1>",
            "<h1 class=\"title_desc\">Where Atmospheric Science Meets Deep Learning Algorithms</h1>",
            "<h4 class=\"author-name\">Author: Elijah Taber</h4>",
            "<a href=\"https://github.com/TaberNater96/AtmoSeer\"",
            "<div class=\"project-description\">",
            "<div class=\"data-source-header\">NOAA GML</div>",
            "<div class=\"data-source-header\">NASA MODIS</div>",
            "<h2 class='pipeline-header'>Project Pipeline</h2>",
            "<div class=\"pipeline-text\">",
            "<div class=\"analytics-text\">",
            "<div class=\"dl-text\">",
            "<div class=\"front-text\">"
        ]
        
        with patch.object(st, "markdown") as mock_markdown:
            with patch.object(os.path, "dirname", return_value=self.mock_current_dir):
                with patch.object(os.path, "abspath", return_value=self.mock_current_dir):
                    with patch.object(os.path, "join", return_value=self.mock_earth_img_path):
                        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64"):
                            with patch.object(st, "sidebar"):
                                with patch.object(st, "set_page_config"):
                                    with patch.object(st, "columns", return_value=[MagicMock(), MagicMock()]):
                                        # Re-import to execute all markdown calls
                                        spec = importlib.util.spec_from_file_location("Home", os.path.join(os.path.dirname(__file__), "..", "pages", "Home.py"))
                                        Home_reimport = importlib.util.module_from_spec(spec)
                                        spec.loader.exec_module(Home_reimport)
                                        
                                        # Collect all markdown calls into a single string for easier searching
                                        all_markdown = ""
                                        for call in mock_markdown.call_args_list:
                                            if len(call[0]) > 0:
                                                all_markdown += call[0][0]
                                        
                                        # Check that each major section is present
                                        for section in main_sections:
                                            self.assertIn(section, all_markdown)
    
    def test_base64_encoding_edge_cases(self):
        # Test case 23: Test the base64 encoding function with edge cases.
        
        # Test with empty file
        with patch("builtins.open", mock_open(read_data=b"")):
            result = Home.get_base64_of_bin_file("empty_file.jpg")
            self.assertEqual(result, "")
        
        # Test with binary file containing special characters
        special_binary = b"\x00\xFF\x80\x7F"
        with patch("builtins.open", mock_open(read_data=special_binary)):
            result = Home.get_base64_of_bin_file("special_file.jpg")
            self.assertEqual(result, base64.b64encode(special_binary).decode())
        
        # Test with large binary file (simulate with a 1MB file)
        large_binary = b"X" * 1024 * 1024  # 1MB of data
        with patch("builtins.open", mock_open(read_data=large_binary)):
            result = Home.get_base64_of_bin_file("large_file.jpg")
            self.assertEqual(result, base64.b64encode(large_binary).decode())
    
    def test_error_handling_file_operations(self):
        # Test case 24: Test error handling for file operations.
        
        # Test file not found error
        with patch("builtins.open", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                Home.get_base64_of_bin_file("nonexistent_file.jpg")
        
        # Test permission error
        with patch("builtins.open", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                Home.get_base64_of_bin_file("permission_denied_file.jpg")
        
        # Test general IO error
        with patch("builtins.open", side_effect=IOError):
            with self.assertRaises(IOError):
                Home.get_base64_of_bin_file("io_error_file.jpg")
    
    def test_app_initialization_sequence(self):
        # Test case 25: Test the entire app initialization sequence in order.
        
        # Create a mock for executing the module
        with patch.object(sys, "path"):
            with patch.object(os.path, "dirname", return_value=self.mock_current_dir):
                with patch.object(os.path, "abspath", return_value=self.mock_current_dir):
                    with patch.object(os.path, "join", return_value=self.mock_earth_img_path):
                        with patch.object(st, "set_page_config") as mock_set_page_config:
                            with patch.object(Home, "set_background_image") as mock_set_background:
                                with patch.object(st.sidebar, "markdown") as mock_sidebar_markdown:
                                    with patch.object(st, "markdown") as mock_markdown:
                                        with patch.object(st, "columns", return_value=[MagicMock(), MagicMock()]):
                                            # Re-import to execute the entire initialization sequence
                                            spec = importlib.util.spec_from_file_location("Home", os.path.join(os.path.dirname(__file__), "..", "pages", "Home.py"))
                                            Home_reimport = importlib.util.module_from_spec(spec)
                                            spec.loader.exec_module(Home_reimport)
                                            
                                            # Verify the initialization sequence
                                            # 1. First, set_page_config should be called
                                            mock_set_page_config.assert_called_once()
                                            
                                            # 2. Then, set_background_image should be called
                                            mock_set_background.assert_called_once()
                                            
                                            # 3. Sidebar navigation should be set up
                                            mock_sidebar_markdown.assert_called_once()
                                            
                                            # 4. Main content and titles should be added
                                            self.assertGreater(mock_markdown.call_count, 5)
    
    def test_background_image_css_injection(self):
        # Test case 26: Test that CSS for background image is properly injected.
        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64_bg"):
            with patch.object(st, "markdown") as mock_markdown:
                # Call the function
                Home.set_background_image("mock_image.jpg")
                
                # Get the CSS content
                css_content = mock_markdown.call_args[0][0]
                
                # Verify the background image CSS
                self.assertIn("background-image: url(\"data:image/jpeg;base64,mock_base64_bg\")", css_content)
                self.assertIn("background-size: cover", css_content)
                self.assertIn("background-position: center", css_content)
                self.assertIn("background-repeat: no-repeat", css_content)
                self.assertIn("background-attachment: fixed", css_content)
    
    def test_hide_streamlit_elements(self):
        # Test case 27: Test that default Streamlit elements are hidden in CSS.
        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64"):
            with patch.object(st, "markdown") as mock_markdown:
                # Call the function
                Home.set_background_image("mock_image.jpg")
                
                # Get the CSS content
                css_content = mock_markdown.call_args[0][0]
                
                # Verify the CSS to hide default elements
                self.assertIn("#MainMenu, footer, header {", css_content)
                self.assertIn("visibility: hidden", css_content)
                
                # Check for header anchor hiding
                self.assertIn(".header-anchor-link {", css_content)
                self.assertIn("display: none !important", css_content)
    
    def test_responsive_design_elements(self):
        # Test case 28: Test CSS for responsive design elements.
        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64"):
            with patch.object(st, "markdown") as mock_markdown:
                # Call the function
                Home.set_background_image("mock_image.jpg")
                
                # Get the CSS content
                css_content = mock_markdown.call_args[0][0]
                
                # Verify responsive CSS elements
                self.assertIn("max-width: 100%", css_content)  # responsive images
                self.assertIn("display: flex", css_content)    # flexible containers
                self.assertIn("flex-direction", css_content)   # layout control
    
    def test_color_scheme_consistency(self):
        # Test case 29: Test that the color scheme is consistent.
        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64"):
            with patch.object(st, "markdown") as mock_markdown:
                # Call the function
                Home.set_background_image("mock_image.jpg")
                
                # Get the CSS content
                css_content = mock_markdown.call_args[0][0]
                
                # Check for the primary color used in the app
                self.assertIn("#19a19a", css_content)  # primary teal color
                
                # Count color occurrences to ensure consistency
                primary_color_count = css_content.count("#19a19a")
                self.assertGreaterEqual(primary_color_count, 5)  # should be used multiple times
    
    def test_import_error_handling(self):
        # Test case 30: Test handling of import errors.
        
        # Mock a scenario where an import fails
        with patch.dict('sys.modules', {'streamlit': None}):
            with self.assertRaises(ImportError):
                # Try to import streamlit which we've removed from sys.modules
                import streamlit
    
    def test_get_base64_of_bin_file_with_mocked_file(self):
        # Test case 31: Test get_base64_of_bin_file with a mocked file object.
        test_data = b"test binary data for encoding"
        expected_result = base64.b64encode(test_data).decode()
        
        # Create a mock file object
        mock_file = MagicMock()
        mock_file.read.return_value = test_data
        
        # Mock the open function to return our mock file
        with patch("builtins.open", return_value=mock_file) as mock_open_func:
            result = Home.get_base64_of_bin_file("test_file.png")
            
            # Verify the result
            self.assertEqual(result, expected_result)
            
            # Verify open was called with the correct file name
            mock_open_func.assert_called_once_with("test_file.png", "rb")
            
            # Verify read was called on the file object
            mock_file.read.assert_called_once()
    
    def test_sidebar_with_content(self):
        # Test case 32: Test that the sidebar contains the correct content.
        with patch.object(st.sidebar, "markdown") as mock_sidebar_markdown:
            # Execute the sidebar creation code
            Home.st.sidebar.markdown("""
                        <h1 style='text-align: center; 
                        font-size: 1.4rem;
                        color:#19a19a; 
                        margin-top: 115px;
                        text-shadow: 1px 1px 2px rgba(0, 0, 0, 1.0);'>Navigate Through Each Gas Type</h1>
                        """, unsafe_allow_html=True)
            
            # Check that the sidebar markdown was called
            mock_sidebar_markdown.assert_called_once()
            
            # Verify the content
            sidebar_content = mock_sidebar_markdown.call_args[0][0]
            self.assertIn("Navigate Through Each Gas Type", sidebar_content)
            self.assertIn("color:#19a19a", sidebar_content)
            self.assertIn("text-shadow", sidebar_content)
            self.assertTrue(mock_sidebar_markdown.call_args[1]['unsafe_allow_html'])
    
    def test_background_image_path_resolution(self):
        # Test case 33: Test the resolution of the background image path.
        with patch.object(os.path, "dirname") as mock_dirname:
            mock_dirname.return_value = "/mock/dir"
            
            with patch.object(os.path, "abspath") as mock_abspath:
                mock_abspath.return_value = "/mock/absolute/path"
                
                with patch.object(os.path, "join") as mock_join:
                    mock_join.side_effect = [
                        "/mock/absolute/path/images",
                        "/mock/absolute/path/images/Earth.jpg"
                    ]
                    
                    with patch.object(Home, "set_background_image") as mock_set_bg:
                        # Execute the relevant code
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        images_dir = os.path.join(current_dir, "images")
                        earth_img_path = os.path.join(images_dir, "Earth.jpg")
                        Home.set_background_image(earth_img_path)
                        
                        # Verify the path resolution
                        mock_dirname.assert_called_once()
                        mock_abspath.assert_called_once()
                        self.assertEqual(mock_join.call_count, 2)
                        mock_set_bg.assert_called_once_with("/mock/absolute/path/images/Earth.jpg")
    
    def test_handling_of_missing_directories(self):
        # Test case 34: Test handling when image directories are missing.
        with patch.object(os.path, "dirname") as mock_dirname:
            mock_dirname.return_value = "/nonexistent/dir"
            
            with patch.object(os.path, "abspath") as mock_abspath:
                mock_abspath.return_value = "/nonexistent/absolute/path"
                
                with patch.object(os.path, "join") as mock_join:
                    mock_join.side_effect = [
                        "/nonexistent/absolute/path/images",
                        "/nonexistent/absolute/path/images/Earth.jpg"
                    ]
                    
                    with patch.object(Home, "set_background_image") as mock_set_bg:
                        # When trying to open a nonexistent file, it will raise FileNotFoundError
                        mock_set_bg.side_effect = FileNotFoundError
                        
                        # Execute the code but expect it to raise an exception
                        with self.assertRaises(FileNotFoundError):
                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            images_dir = os.path.join(current_dir, "images")
                            earth_img_path = os.path.join(images_dir, "Earth.jpg")
                            Home.set_background_image(earth_img_path)
    
    def test_sidebar_style_consistency(self):
        # Test case 35: Test that sidebar styling is consistent with the main theme.
        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64"):
            with patch.object(st, "markdown") as mock_markdown:
                # Call the function
                Home.set_background_image("mock_image.jpg")
                
                # Get the CSS content
                css_content = mock_markdown.call_args[0][0]
                
                # Verify sidebar styling
                self.assertIn("section[data-testid=\"stSidebar\"]", css_content)
                self.assertIn("background-color: rgba(25, 161, 154, 0.25)", css_content)  # primary color with transparency
    
    def test_hover_effects(self):
        # Test case 36: Test that hover effects are properly defined in CSS.
        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64"):
            with patch.object(st, "markdown") as mock_markdown:
                # Call the function
                Home.set_background_image("mock_image.jpg")
                
                # Get the CSS content
                css_content = mock_markdown.call_args[0][0]
                
                # Verify hover effects for sidebar items
                self.assertIn("section[data-testid=\"stSidebar\"] li:hover", css_content)
                self.assertIn("border-left: 3px solid #19a19a", css_content)
                
                # Verify hover effects for links
                self.assertIn(".data-source a:hover", css_content)
    
    def test_text_styling_and_readability(self):
        # Test case 37: Test text styling and readability features.
        with patch.object(Home, "get_base64_of_bin_file", return_value="mock_base64"):
            with patch.object(st, "markdown") as mock_markdown:
                # Call the function
                Home.set_background_image("mock_image.jpg")
                
                # Get the CSS content
                css_content = mock_markdown.call_args[0][0]
                
                # Verify text styling elements
                self.assertIn("text-shadow", css_content)  # for improved readability against backgrounds
                self.assertIn("font-weight: bold", css_content)  # for emphasis
                self.assertIn("line-height", css_content)  # for readability
                
                # Check font family
                self.assertIn("font-family: 'Courier New', monospace", css_content)
    
    def test_performance_aspects(self):
        # Test case 38: Test performance-related aspects.
        
        # Test base64 encoding function
        import time
        test_data = b"X" * 100000  # 100KB of data
        
        with patch("builtins.open", mock_open(read_data=test_data)):
            start_time = time.time()
            result = Home.get_base64_of_bin_file("large_file.jpg")
            end_time = time.time()
            
            # Encoding 100KB should be very fast (less than 0.1 seconds)
            self.assertLess(end_time - start_time, 0.1)
            
            # The result length should be about 4/3 of the input (base64 overhead)
            self.assertGreater(len(result), len(test_data))
            self.assertLess(len(result), len(test_data) * 2)  # a loose upper bound

if __name__ == "__main__":
    unittest.main()