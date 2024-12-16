from typing import List
import pandas as pd
import requests
import io

class GMLDataExtractor:
    """
    A class to fetch and process greenhouse gas data from the GML repository.
    """
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.data_frames = []
    
    def fetch_and_process(self) -> pd.DataFrame:
        """
        Fetches and processes data from each URL in the urls attribute.
        Processes data by identifying the header end, converting text to DataFrame,
        and appending to the data_frames list. Combines all data frames into one
        upon completion.

        Returns:
            pd.DataFrame: The combined DataFrame of all processed data. Returns an
            empty DataFrame if no data was processed.
        """
        for url in self.urls:
            response = requests.get(url)
            # If the response is successful, strip the header and add the data to the dataframe list
            if response.status_code == 200:
                data = response.text
                header_end_idx = self.find_header_end(data)
                df = self.text_to_dataframe(data, header_end_idx)
                self.data_frames.append(df)
            else:
                print(f"Failed to fetch data from {url}")
        
        print("Done.")
        
        # After the API calls complete, combine all the dataframes into a single master dataframe
        if self.data_frames:
            combined_df = pd.concat(self.data_frames, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
        
    def fetch_and_process_co2_data(self) -> pd.DataFrame:
        print("Processing CO2 Data...")
        return self.fetch_and_process()

    def fetch_and_process_ch4_data(self) -> pd.DataFrame:
        print("Processing CH4 Data...")
        return self.fetch_and_process()

    def fetch_and_process_sf6_data(self) -> pd.DataFrame:
        print("Processing SF6 Data...")
        return self.fetch_and_process()

    def fetch_and_process_n2o_data(self) -> pd.DataFrame:
        print("Processing N2O Data...")
        return self.fetch_and_process()

    def find_header_end(self, text: str) -> int:
        """
        Determines the index where actual data starts in the fetched text.

        Parameters:
            text (str): The complete raw text fetched from a URL.

        Returns:
            int: The line index where the header ends and the actual data begins.
        """
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#'): # the actual tabular data starts the line after #
                return i
        # Default return in case no data line is found
        return len(lines)  

    def text_to_dataframe(self, text: str, header_end_idx: int) -> pd.DataFrame:
        """
        Converts text data into a pandas DataFrame starting from a specified index.

        Parameters:
            text (str): The complete raw text fetched from a URL.
            header_end_idx (int): The index at which to start reading the data into a DataFrame.

        Returns:
            pd.DataFrame: The DataFrame created from the text data.
        """
        data_str = '\n'.join(text.split('\n')[header_end_idx:])
        df = pd.read_csv(io.StringIO(data_str), sep="\s+")
        return df