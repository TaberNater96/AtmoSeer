import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import earthaccess
import xarray as xr
from typing import Dict, Tuple
import logging
from tqdm.notebook import tqdm
import re

class ModisProcessor:
    """
    A specialized processor that leverages NASA's MODIS satellite data to estimate vegetation biomass density around 
    greenhouse gas measurement sites, providing crucial context for carbon sink capacity. By analyzing MODIS-derived
    vegetation indices within configurable buffer zones around measurement locations, it allows the quantification of 
    local biological carbon sequestration potential that may influence greenhouse gas readings. The processor implements 
    temporal interpolation to generate continuous monthly biomass estimates from seasonal measurements, accounting for 
    vegetation cycles and geographical variations including polar regions. This integration of satellite-derived vegetation 
    data with greenhouse gas measurements helps create a more complete understanding of local carbon flux dynamics, particularly 
    useful for models predicting greenhouse gas concentrations and studying the relationship between terrestrial ecosystems and 
    atmospheric composition.
    """
    def __init__(
        self, 
        username: str, 
        password: str, 
        buffer_radius_km: float = 50
    ):
        """
        Initialize the ModisProcessor for retrieving and processing MODIS satellite data.
        
        Creates a new ModisProcessor instance that handles authentication, logging setup,and defines the geographical 
        buffer zone for data collection. The processor uses a local cache directory to store intermediate results and 
        optimize repeated queries.
        
        Parameters
        ----------
        username : str
            NASA Earthdata login username
        password : str
            NASA Earthdata login password
        buffer_radius_km : float, optional
            Radius in kilometers around each site to collect MODIS data (default: 50)
            
        Attributes
        ----------
        cache_dir : Path
            Directory for storing cached MODIS data processing results
        buffer_radius_km : float
            Radius of the data collection zone in kilometers
        logger : Logger
            Configured logging instance for the processor
        """
        self.setup_logging()
        self.setup_auth(username, password)
        self.buffer_radius_km = buffer_radius_km
        self.cache_dir = Path("modis_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        """
        Configure the logging system for the ModisProcessor.
        
        Sets up basic logging configuration at INFO level for tracking processing status, warnings, and errors during 
        MODIS data retrieval and processing.
        
        Attributes
        ----------
        logger : Logger
            Configured logging instance stored in self.logger
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_auth(
        self, 
        username: str, 
        password: str
    ):
        """
        Set up NASA Earthdata authentication using environment variables.
        
        Configures authentication for accessing NASA's Earth Observing System Data and Information System (EOSDIS) 
        using the earthaccess library. Credentials are stored as OS environment variables for secure access.
        
        Parameters
        ----------
        username : str
            NASA Earthdata login username
        password : str
            NASA Earthdata login password
            
        Notes
        -----
        Uses the earthaccess library's environment strategy for authentication, which stores credentials as 
        environment variables:
        - EARTHDATA_USERNAME
        - EARTHDATA_PASSWORD
        """
        os.environ['EARTHDATA_USERNAME'] = username
        os.environ['EARTHDATA_PASSWORD'] = password
        earthaccess.login(strategy='environment')

    def _get_bbox(
        self, 
        lat: float, 
        lon: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate a bounding box around a given latitude/longitude coordinate.
        
        Creates a square buffer zone around a geographical point using the instance's buffer_radius_km. The calculation 
        accounts for the Earth's spherical geometry by adjusting the longitude buffer based on the cosine of latitude, as
        degrees of longitude become smaller (in km) as you move away from the equator.
        
        The returned bounding box format is (min_lon, min_lat, max_lon, max_lat), which is compatible with most geospatial 
        APIs including the NASA MODIS data service.
        
        Parameters
        ----------
        lat : float
            Center latitude in decimal degrees (-90 to 90)
        lon : float
            Center longitude in decimal degrees (-180 to 180)
            
        Returns
        -------
        Tuple[float, float, float, float]
            Bounding box coordinates in format (min_lon, min_lat, max_lon, max_lat)
        
        Notes
        -----
        Uses the approximation of 111.32 km per degree at the equator. The longitude buffer is adjusted by cos(latitude) to 
        account for the convergence of meridians at higher latitudes. Since Earth is a sphere, longitudinal distances converge
        as the distance to the poles decreases. This means that the actual distance between longitude degrees chnages based on
        what the actual latitude is.
        
        * At equator (0°): 111.32 km * cos(0°) = 111.32 km per degree
        * At 60° latitude: 111.32 km * cos(60°) = 55.66 km per degree
        * At 89° latitude: 111.32 km * cos(89°) = 1.94 km per degree
        """
        # Standard distance of 1 degree latitude at equator in km
        km_per_degree = 111.32
        
        # Calculate latitude buffer (constant in degrees)
        lat_buffer = self.buffer_radius_km / km_per_degree
        
        # Calculate longitude buffer (varies with latitude), cos(lat) adjustment accounts for meridian convergence
        lon_buffer = self.buffer_radius_km / (km_per_degree * np.cos(np.deg2rad(lat))) # numpy expects radians
        
        return (lon - lon_buffer, lat - lat_buffer, lon + lon_buffer, lat + lat_buffer)

    def _parse_modis_filename(
        self, 
        url: str
    ) -> datetime:
        """
        Extract the acquisition date from a MODIS product filename.
        
        MODIS filenames follow a standardized naming convention where the acquisition
        date is encoded using a year and day-of-year format. This method extracts
        these components using regex and converts them to a Python datetime object.
        
        The expected filename pattern contains '.AYYYYDDD.' where:
        - YYYY is the 4-digit year
        - DDD is the 3-digit day of year (001-366)
        
        Parameters
        ----------
        url : str
            Full URL or filename of MODIS product
            
        Returns
        -------
        datetime or None
            Datetime object representing the acquisition date if parsing successful,
            None if the filename doesn't match the expected pattern
        """
        # Extract just the filename from the full URL path
        filename = url.split('/')[-1]
        
        # Extract year and day of year
        match = re.search(r'\.A(\d{4})(\d{3})\.', filename)
        
        if match:
            year, doy = map(int, match.groups()) # where doy is "day of year"
            # Convert year and day-of-year to datetime
            return datetime.strptime(f"{year}-{doy}", "%Y-%j")
        return None
    
    def _interpolate_monthly(
        self, 
        seasonal_values: list
    ) -> Dict[int, float]:
        """
        Interpolate monthly biomass values from sparse seasonal measurements.
        
        Uses linear interpolation to estimate biomass density values for all months based on available seasonal measurements. 
        The interpolation treats the months as equally spaced points and performs linear interpolation between them to
        generate a complete set of monthly values. If no seasonal values are provided, falls back to default values to ensure
        the method always returns a complete set of monthly data.
        
        Parameters
        ----------
        seasonal_values : list
            List of tuples containing (month, biomass_value) pairs from seasonal measurements. Month should be 1-12, and 
            values are biomass density measurements from MODIS NDVI data.
            
        Returns
        -------
        Dict[int, float]
            Dictionary mapping each month (1-12) to its interpolated biomass value. Returns default values if seasonal_values 
            is empty.
        
        Notes
        -----
        The interpolation uses numpy's interp function which performs linear interpolation between the provided points. 
        Values are interpolated across all 12 months regardless of how many seasonal measurements are provided.
        """
        if not seasonal_values:
            return self._get_default_values()
            
        seasonal_values.sort(key=lambda x: x[0])
        
        months = [m for m, _ in seasonal_values]
        values = [v for _, v in seasonal_values]
        
        # Generate values for all months using linear interpolation
        all_months = np.arange(1, 13)
        interp_values = np.interp(all_months, months, values)
        
        return {month: value for month, value in zip(all_months, interp_values)}
    
    def _get_default_values(self) -> Dict[int, float]:
        """
        Generate default monthly biomass density values.
        
        Creates a sinusoidal pattern of biomass density values throughout the year, simulating a typical seasonal 
        vegetation cycle. The pattern follows:
        
        - Base value of 50 (representing medium vegetation density)
        - Amplitude of 10 (representing seasonal variation)
        - Peak in summer months, trough in winter months
        
        The sine wave is shifted to align peaks with typical Northern Hemisphere growing seasons.
        
        Returns
        -------
        Dict[int, float]
            Dictionary mapping months (1-12) to default biomass values following a sinusoidal pattern. Values range 
            approximately from 40 to 60.
        """
        base = 50       # base biomass density value
        amplitude = 10  # seasonal variation magnitude
        
        # Generate sinusoidal pattern for months 1-12. Phase shift aligns with typical hemisphere seasons
        return {
            month: base + amplitude * np.sin((month - 1) * np.pi / 6)
            for month in range(1, 13)
        }

    def process_site(self, site: str, lat: float, lon: float) -> Dict[int, float]:
        """
        Process MODIS satellite data for a specific geographical site to calculate monthly biomass density values.
        
        This method handles the complete workflow for retrieving and processing MODIS vegetation index data 
        (MOD13Q1 product) for a given site location, while using a local cache to store processed results. It
        handles sites near the poles with default values due to satellite coverage limitations. This process 
        will fall back to default values if data retrieval or processing fails. The workflow follows:
        
        1. Checking for cached results to avoid redundant processing
        2. Handling special cases for near-polar sites by validating latitude
        3. Query MODIS within a buffered geographical region by using the bounding box
        4. Group and sort the granules by month
        5. Processing temporal data to extract seasonal vegetation patterns
        6. Interpolating monthly values from seasonal measurements
        7. Cache the results
        
        If any step fails, the method falls back to default values rather than failing completely.
        
        The biomass density values are derived from NDVI (Normalized Difference Vegetation Index) measurements, 
        scaled and adjusted to represent relative biomass density. The scaling formula converts NDVI (-1 to 1 scale) 
        to biomass density using: (NDVI + 1) * 50
        
        In remote sensing and satellite data, granules are standardized chunks or subsets of data that represent specific 
        geographic regions and time periods, typically organized as individual files to make large datasets more manageable. 
        The NASA MODIS satellite takes pictures of Earth every 1-2 days, where each picture is processed into a granule. MOD13Q1
        is a 16-day composite product, where each granule represents a 16-day period.
        
        Parameters
        ----------
        site : str
            Identifier for the measurement site
        lat : float
            Site latitude in decimal degrees (-90 to 90)
        lon : float
            Site longitude in decimal degrees (-180 to 180)
            
        Returns
        -------
        Dict[int, float]
            Dictionary mapping months (1-12) to estimated biomass density values
            
        Raises
        ------
        Exception
            Propagates exceptions from data download or processing, though includes fallback to default values for error cases
            
        Notes
        -----
        - Caches results in '{site}_biomass.csv' files for future use
        - Uses a fixed date range (2020-2023) for historical MODIS data
        - Samples one month per season (January, April, July, October)
        - Near-polar sites (|latitude| > 80°) automatically use default values
        - Default values follow a sinusoidal pattern with base=50, amplitude=10
        """
        cache_file = self.cache_dir / f"{site}_biomass.csv"
        
        # Check if there is cached results for this site
        if cache_file.exists():
            self.logger.info(f"Using cached data for site {site}")
            return pd.read_csv(cache_file, index_col=0).to_dict()['biomass']
        
        # Handle polar regions where MODIS data may be unreliable
        if abs(lat) > 80:
            self.logger.info(f"Site {site} near pole, using default values")
            return self._get_default_values()
            
        bbox = self._get_bbox(lat, lon)
        try:
            # Search for data within specific temporal bounds
            start_date = '2020-01-01'
            end_date = '2023-12-31'
            
            # Query MODIS vegetation index product (MOD13Q1)
            results = earthaccess.search_data(
                short_name='MOD13Q1',  # the MODIS vegetation indices 16-day L3 product
                cloud_hosted=True,      # use NASA cloud-optimized data access
                temporal=(start_date, end_date),
                bounding_box=bbox
            )
            
            if not results:
                self.logger.warning(f"No data found for site {site}")
                return self._get_default_values()
            
            self.logger.info(f"\nProcessing site {site} with {len(results)} granules found")
            
            # Organize granules by month for temporal analysis
            monthly_granules = {}
            
            # Sort granules by date and keep most recent for each month
            for granule in results:
                links = granule.data_links()
                if not links:
                    continue
                    
                date = self._parse_modis_filename(links[0])
                if date:
                    monthly_granules.setdefault(date.month, []).append((date, granule))
            
            # Sort by date and keep most recent for each month
            for month in monthly_granules:
                monthly_granules[month].sort(key=lambda x: x[0], reverse=True)
            
            seasonal_values = []
            
            # For each season, process the most recent granule
            for month in [1, 4, 7, 10]:  # representative months for each season
                if month in monthly_granules:
                    try:
                        _, granule = monthly_granules[month][0]  # Most recent granule
                        data = earthaccess.download(granule)
                        
                        # Process the granule if it is available
                        if data:
                            # Extract and process NDVI data
                            ds = xr.open_dataset(data[0], engine='netcdf4')
                            ndvi = ds['250m 16 days NDVI'].mean().item()
                            # Convert NDVI to biomass density estimate
                            seasonal_values.append((month, (ndvi + 1) * 50))
                            self.logger.info(f"Successfully processed {site} for month {month}")
                        else:
                            self.logger.warning(f"No data downloaded for site {site} month {month}")
                    except Exception as e:
                        self.logger.warning(f"Error processing month {month} for site {site}: {str(e)}")
                else:
                    self.logger.warning(f"No granules found for site {site} month {month}")
            
            if not seasonal_values:
                self.logger.warning(f"No valid data for site {site}, using default values")
                return self._get_default_values()
            
            # Interpolate monthly values from seasonal measurements
            monthly_values = self._interpolate_monthly(seasonal_values)
            
            # Cache results for future use
            pd.DataFrame.from_dict(
                monthly_values, 
                orient='index',
                columns=['biomass']
            ).to_csv(cache_file)
            
            return monthly_values
            
        except Exception as e:
            self.logger.error(f"Error processing site {site}: {str(e)}")
            return self._get_default_values() # fallback to default values

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process MODIS satellite data for all unique sites in a DataFrame to add biomass density values.
        
        This method orchestrates bulk processing of MODIS vegetation data across multiple measurement sites. The 
        workflow process is as follows:
        
        - Creates a copy of input data to preserve the original structure
        - Identifies and processes each unique site once through the process_site method
        - Uses cached results when available to optimize processing time
        - Maps monthly biomass values back to the original temporal granularity
        - Provides progress tracking and error handling for each site
        - Maintains processing visibility through tqdm progress bars
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing at minimum these columns:
            - 'site': Site identifier
            - 'latitude': Site latitude
            - 'longitude': Site longitude
            Must also have a datetime index or date column that can be used to determine months for biomass mapping
            
        Returns
        -------
        pd.DataFrame
            Copy of input DataFrame with additional 'biomass_density' column.
            The biomass values are mapped from monthly calculations to match
            the temporal resolution of the input data.
        
        Warning
        -------
        Large DataFrames with many unique sites may take significant processing time due to MODIS data retrieval and 
        processing for each unique location. Consider pre-processing and caching results for large datasets.
        """
        # Create a copy to avoid modifying the input DataFrame
        result_df = df.copy()
        result_df['biomass_density'] = np.nan
        
        # Extract unique sites with their coordinates
        sites = df.groupby('site')[['latitude', 'longitude']].first().reset_index()
        total_sites = len(sites)
        
        print(f"\nProcessing {total_sites} unique sites")
        
        # Process each site with progress tracking
        for _, site_row in tqdm(
            sites.iterrows(), 
            total=total_sites,
            desc="Processing sites",
            position=0
        ):
            site = site_row['site']
            
            # Get monthly biomass values for this site
            monthly_values = self.process_site(
                site, 
                site_row['latitude'], 
                site_row['longitude']
            )
            
            # Map monthly values back to the original temporal resolution
            site_mask = result_df['site'] == site
            result_df.loc[site_mask, 'biomass_density'] = result_df.loc[site_mask].index.month.map(monthly_values)
        
        return result_df
        
    def cleanup(self):
        """
        Remove all downloaded MODIS data files and cached processing results.
        
        Performs cleanup operations by removing:
        1. The cache directory containing processed biomass values
        2. The data directory containing downloaded MODIS granules
        
        Raises
        ------
        Exception
            Any errors during directory removal are logged but not re-raised to ensure cleanup attempts continue even 
            if one step fails
        
        Notes
        -----
        - Attempts to remove both directories even if one fails
        - Logs success or failure of each cleanup operation
        - Safe to call even if directories don't exist
        """
        try:
            shutil.rmtree(self.cache_dir)
            self.logger.info("Cleaned up cache directory")
            
            data_dir = Path("data")
            if data_dir.exists():
                shutil.rmtree(data_dir)
                self.logger.info("Cleaned up data directory")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

def add_biomass_density(df: pd.DataFrame, 
                       earthdata_username: str, 
                       earthdata_password: str) -> pd.DataFrame:
    """
    Add biomass density estimates to a DataFrame using MODIS satellite data.
    
    This is the main entry point function that wraps the ModisProcessor class functionality. It handles the workflow of:
    
    1. Creating a ModisProcessor instance
    2. Processing the input DataFrame
    3. Cleaning up temporary files
    4. Makes sure cleanup occurs even if processing fails
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at minimum:
        - 'site' column with site identifiers
        - 'latitude' column with site latitudes
        - 'longitude' column with site longitudes
    earthdata_username : str
        NASA Earthdata login username
    earthdata_password : str
        NASA Earthdata login password
        
    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional 'biomass_density' column containing estimated biomass values derived from 
        MODIS vegetation indices
        
    Raises
    ------
    Exception
        Re-raises any exceptions from processing after ensuring cleanup is performed
    """
    processor = ModisProcessor(earthdata_username, earthdata_password)
    
    try:
        result_df = processor.process_dataframe(df)
        return result_df
    except Exception as e:
        processor.cleanup()
        raise e
    finally:
        processor.cleanup()