import time
import boto3
from typing import List, Dict, Any
from IPython.display import display, clear_output

class DynamoDBDataLoader:
    def __init__(self, region_name: str = 'us-west-2'):
        """
        Initialize the DynamoDB resource.

        Args:
            region_name (str): AWS region where the DynamoDB resource is located. Defaults to 'us-west-2'.
        """
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)

    def prepare_data(self, row: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepare a single row of data for DynamoDB, converting necessary fields to strings to meet DynamoDB requirements.

        Args:
            row (Dict[str, Any]): A dictionary representing a row of data with keys matching DynamoDB table's column names.

        Returns:
            Dict[str, str]: A dictionary with all values converted to strings, ready for DynamoDB insertion.
        """
        return {
            'datetime': str(row['datetime']),
            'site': str(row['site']),
            'ppm': str(row['ppm']),
            'latitude': str(row['latitude']),
            'longitude': str(row['longitude']),
            'altitude': str(row['altitude']),
            'elevation': str(row['elevation']),
            'intake_height': str(row['intake_height']),
            'qcflag': str(row['qcflag']),
            'year': str(row['year']),
            'month': str(row['month']),
            'day': str(row['day']),
            'season': str(row['season']),
            'co2_change_rate': str(row['co2_change_rate']), # change to specific gas name before loading
            'gas': str(row['gas'])
        }

    def batch_write_items(self, table_name: str, data: List[Dict[str, str]], sleep_time: float = 1.00):
        """
        Write batches of prepared data to a specified DynamoDB table with a sleep interval to control the write throughput.
        This method is specifically designed to stay within the DynamoDB free tier.

        Args:
            table_name (str): Name of the DynamoDB table to which data is written.
            data (List[Dict[str, str]]): List of dictionaries where each dictionary is a record to be inserted into the table.
            sleep_time (float): Time in seconds to wait between batches to avoid throttling. Defaults to 1.00.
        """
        table = self.dynamodb.Table(table_name)
        batch_size = 22
        total_batches = (len(data) + batch_size - 1) // batch_size  # calculate the total number of batches
        display("Starting data upload to DynamoDB...")

        for batch_number in range(total_batches):
            start_index = batch_number * batch_size
            batch = data[start_index:start_index + batch_size]
            with table.batch_writer() as writer:
                for item in batch:
                    writer.put_item(Item=item)
            time.sleep(sleep_time)

            # Calculate and display the progress
            progress = ((batch_number + 1) / total_batches) * 100
            clear_output(wait=True)  # clear the previous output before displaying the new progress
            display(f"Upload progress: {progress:.2f}% completed.")

        clear_output(wait=True)
        display("Data upload completed.")