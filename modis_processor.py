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
from datetime import datetime, timedelta

class ModisProcessor:
    def __init__(self, username: str, password: str, buffer_radius_km: float = 50):
        self.setup_logging()
        self.setup_auth(username, password)
        self.buffer_radius_km = buffer_radius_km
        self.cache_dir = Path("modis_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_auth(self, username: str, password: str):
        os.environ['EARTHDATA_USERNAME'] = username
        os.environ['EARTHDATA_PASSWORD'] = password
        earthaccess.login(strategy='environment')

    def _get_bbox(self, lat: float, lon: float) -> Tuple[float, float, float, float]:
        """Calculate bounding box for coordinate."""
        km_per_degree = 111.32
        lat_buffer = self.buffer_radius_km / km_per_degree
        lon_buffer = self.buffer_radius_km / (km_per_degree * np.cos(np.deg2rad(lat)))
        return (lon - lon_buffer, lat - lat_buffer, lon + lon_buffer, lat + lat_buffer)

    def _parse_modis_filename(self, url: str) -> datetime:
        """Extract date from MODIS filename."""
        filename = url.split('/')[-1]
        match = re.search(r'\.A(\d{4})(\d{3})\.', filename)
        if match:
            year, doy = map(int, match.groups())
            return datetime.strptime(f"{year}-{doy}", "%Y-%j")
        return None

    def process_site(self, site: str, lat: float, lon: float) -> Dict[int, float]:
        """Process a single site and return monthly biomass values."""
        cache_file = self.cache_dir / f"{site}_biomass.csv"
        
        if cache_file.exists():
            self.logger.info(f"Using cached data for site {site}")
            return pd.read_csv(cache_file, index_col=0).to_dict()['biomass']
        
        if abs(lat) > 80:
            self.logger.info(f"Site {site} near pole, using default values")
            return self._get_default_values()
            
        bbox = self._get_bbox(lat, lon)
        try:
            # Search for data with wider temporal range
            start_date = '2020-01-01'
            end_date = '2023-12-31'
            
            results = earthaccess.search_data(
                short_name='MOD13Q1',
                cloud_hosted=True,
                temporal=(start_date, end_date),
                bounding_box=bbox
            )
            
            if not results:
                self.logger.warning(f"No data found for site {site}")
                return self._get_default_values()
            
            self.logger.info(f"\nProcessing site {site} with {len(results)} granules found")
            
            # Group granules by month
            monthly_granules = {}
            for granule in results:
                links = granule.data_links()
                if not links:
                    continue
                    
                date = self._parse_modis_filename(links[0])
                if date:
                    monthly_granules.setdefault(date.month, []).append((date, granule))
            
            # Sort granules within each month by date and keep the most recent
            for month in monthly_granules:
                monthly_granules[month].sort(key=lambda x: x[0], reverse=True)
            
            seasonal_values = []
            for month in [1, 4, 7, 10]:  # One month per season
                if month in monthly_granules:
                    try:
                        # Use the most recent granule for this month
                        _, granule = monthly_granules[month][0]
                        data = earthaccess.download(granule)
                        
                        if data:
                            ds = xr.open_dataset(data[0], engine='netcdf4')
                            ndvi = ds['250m 16 days NDVI'].mean().item()
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
            
            monthly_values = self._interpolate_monthly(seasonal_values)
            
            # Save to cache
            pd.DataFrame.from_dict(monthly_values, 
                                orient='index',
                                columns=['biomass']).to_csv(cache_file)
            
            return monthly_values
            
        except Exception as e:
            self.logger.error(f"Error processing site {site}: {str(e)}")
            return self._get_default_values()
    
    def _interpolate_monthly(self, seasonal_values: list) -> Dict[int, float]:
        """Interpolate monthly values from seasonal measurements."""
        if not seasonal_values:
            return self._get_default_values()
            
        seasonal_values.sort(key=lambda x: x[0])
        months = [m for m, _ in seasonal_values]
        values = [v for _, v in seasonal_values]
        
        all_months = np.arange(1, 13)
        interp_values = np.interp(all_months, months, values)
        
        return {month: value for month, value in zip(all_months, interp_values)}
    
    def _get_default_values(self) -> Dict[int, float]:
        """Return default monthly biomass values."""
        base = 50
        amplitude = 10
        return {
            month: base + amplitude * np.sin((month - 1) * np.pi / 6)
            for month in range(1, 13)
        }

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire DataFrame and add biomass density column."""
        result_df = df.copy()
        result_df['biomass_density'] = np.nan
        
        sites = df.groupby('site')[['latitude', 'longitude']].first().reset_index()
        total_sites = len(sites)
        
        print(f"\nProcessing {total_sites} unique sites")
        
        for _, site_row in tqdm(sites.iterrows(), 
                            total=total_sites,
                            desc="Processing sites",
                            position=0):
            site = site_row['site']
            monthly_values = self.process_site(
                site, 
                site_row['latitude'], 
                site_row['longitude']
            )
            
            site_mask = result_df['site'] == site
            result_df.loc[site_mask, 'biomass_density'] = \
                result_df.loc[site_mask].index.month.map(monthly_values)
        
        return result_df
        
    def cleanup(self):
        """Remove downloaded files and cache."""
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
    """Add biomass density feature to DataFrame."""
    processor = ModisProcessor(earthdata_username, earthdata_password)
    
    try:
        result_df = processor.process_dataframe(df)
        processor.cleanup()
        return result_df
    except Exception as e:
        processor.cleanup()
        raise e