import pandas as pd
from typing import Union, Dict, List
from datetime import datetime, date
import numpy as np
from dataclasses import dataclass

@dataclass
class MeasurementRecord:
    """Represents a single gas measurement with location data."""
    date: date
    site: str
    ppm: float
    latitude: float
    longitude: float
    altitude: float

class NOAALookup:
    """
    Handles lookups of historical NOAA gas measurements.
    Supports both single date and date range queries with location data.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize lookup with preprocessed NOAA data.
        
        Args:
            data: DataFrame containing gas measurements with required columns:
                 date, site, ppm, latitude, longitude, altitude
        """
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date']).dt.date
        self._validate_data()
        
        # Store date range info
        self.earliest_date = min(self.data['date'])
        self.latest_date = max(self.data['date'])
        
    def _validate_data(self) -> None:
        """Ensure DataFrame has required columns."""
        required_columns = ['date', 'site', 'ppm', 'latitude', 'longitude', 'altitude']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _find_nearest_date(self, target_date: date) -> date:
        """Find the closest date in the dataset to the target date."""
        available_dates = self.data['date'].unique()
        nearest_date = min(available_dates, key=lambda x: abs((x - target_date).days))
        return nearest_date
    
    def _date_to_record(self, row: pd.Series) -> MeasurementRecord:
        """Convert a DataFrame row to a MeasurementRecord."""
        return MeasurementRecord(
            date=row['date'],
            site=row['site'],
            ppm=row['ppm'],
            latitude=row['latitude'],
            longitude=row['longitude'],
            altitude=row['altitude']
        )
    
    def lookup_date(
        self, 
        query_date: Union[str, date, datetime],
        as_dict: bool = False
    ) -> Union[List[MeasurementRecord], List[Dict]]:
        """
        Look up all measurements for a specific date.
        
        Args:
            query_date: Date to look up measurements for
            as_dict: If True, return records as dictionaries instead of MeasurementRecord objects
            
        Returns:
            List of measurements from all sites for the nearest available date
        """
        # Convert query date to date object
        if isinstance(query_date, str):
            query_date = pd.to_datetime(query_date).date()
        elif isinstance(query_date, datetime):
            query_date = query_date.date()
            
        # Validate date range
        if query_date < self.earliest_date:
            raise ValueError(
                f"Date {query_date} is before earliest available date ({self.earliest_date})"
            )
        if query_date > self.latest_date:
            raise ValueError(
                f"Date {query_date} is after latest available date ({self.latest_date})"
            )
            
        # Find nearest date with data
        nearest_date = self._find_nearest_date(query_date)
        
        # Get all measurements for that date
        records = self.data[self.data['date'] == nearest_date]
        
        if not records.empty:
            if query_date != nearest_date:
                print(f"Note: No data for {query_date}. Using nearest available date: {nearest_date}")
                
            measurements = [self._date_to_record(row) for _, row in records.iterrows()]
            
            if as_dict:
                return [vars(record) for record in measurements]
            return measurements
        else:
            return []
    
    def lookup_range(
        self,
        start_date: Union[str, date, datetime],
        end_date: Union[str, date, datetime],
        as_dict: bool = False
    ) -> Dict[date, Union[List[MeasurementRecord], List[Dict]]]:
        """
        Look up all measurements within a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            as_dict: If True, return records as dictionaries instead of MeasurementRecord objects
            
        Returns:
            Dictionary mapping dates to lists of measurements
        """
        # Convert dates to date objects
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()
            
        # Validate date range
        if start_date < self.earliest_date:
            raise ValueError(
                f"Start date {start_date} is before earliest available date ({self.earliest_date})"
            )
        if end_date > self.latest_date:
            raise ValueError(
                f"End date {end_date} is after latest available date ({self.latest_date})"
            )
        if end_date < start_date:
            raise ValueError("End date must be after start date")
            
        # Get all measurements within range
        mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
        records = self.data[mask]
        
        # Group by date
        results = {}
        for date_val, group in records.groupby('date'):
            measurements = [self._date_to_record(row) for _, row in group.iterrows()]
            if as_dict:
                results[date_val] = [vars(record) for record in measurements]
            else:
                results[date_val] = measurements
                
        return results
    
    def get_available_sites(self) -> List[Dict[str, Union[str, float]]]:
        """Get information about all measurement sites in the dataset."""
        site_info = []
        for _, group in self.data.groupby('site'):
            first_record = group.iloc[0]
            site_info.append({
                'site': first_record['site'],
                'latitude': first_record['latitude'],
                'longitude': first_record['longitude'],
                'altitude': first_record['altitude'],
                'date_range': f"{min(group['date'])} to {max(group['date'])}"
            })
        return site_info