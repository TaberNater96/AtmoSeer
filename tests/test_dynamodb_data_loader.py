import unittest
from unittest.mock import patch, MagicMock
from ..etl.\
            load.\
                dynamodb_data_loader import DynamoDBDataLoader

class TestDynamoDBDataLoader(unittest.TestCase):
    """
    Provides a high-level interface for loading data into AWS DynamoDB tables. This class includes methods for
    preparing data entries to meet DynamoDB's format requirements and for efficiently writing large batches of data
    with controlled throughput to avoid hitting provisioned write limits. It simplifies interactions with DynamoDB
    by abstracting complex batching and data preparation logic.
    """
    
    def setUp(self) -> None:
        """
        Set up the test environment before each test.
        Initializes the DynamoDBDataLoader with a mock DynamoDB resource.
        """
        self.loader = DynamoDBDataLoader(region_name='us-west-2')

    # Patch Decorator: controls the behavior of external dependencies by temporarily replacing the object specified by its 
    # target with a 'Mock' or another object during the test and restores the original object after the test is done. It modifies 
    # the target object for the duration of the test and automatically reverts it back to its original state at the end of the test.
    @patch('boto3.resource') 
    def test_initialization(self, mock_boto3_resource: MagicMock) -> None:
        """
        Test the initialization of the DynamoDB resource within the DynamoDBDataLoader class.

        Args:
            mock_boto3_resource (MagicMock): A mock of the boto3.resource function.

        Asserts:
            Ensures that boto3.resource is called exactly once with the specified parameters.
        """
        DynamoDBDataLoader(region_name='us-west-2')
        mock_boto3_resource.assert_called_once_with('dynamodb', region_name='us-west-2')

    def test_prepare_data(self) -> None:
        """
        Test the prepare_data function to ensure it correctly formats each data row for DynamoDB.

        Asserts:
            Compares the result of prepare_data with a manually prepared expected dictionary,
            ensuring all data types are correctly converted to strings.
        """
        sample_row = {
            'datetime': '2021-01-01T00:00:00Z',
            'site': 'TestSite',
            'ppm': '400',
            'latitude': '45.0',
            'longitude': '-122.0',
            'altitude': '200',
            'elevation': '150',
            'intake_height': '10',
            'qcflag': '...',
            'year': '2021',
            'month': '1',
            'day': '1',
            'season': 'Winter',
            'co2_change_rate': '0.1',
            'gas': 'CO2'
        }
        expected = {k: str(v) for k, v in sample_row.items()}
        result = self.loader.prepare_data(sample_row)
        self.assertEqual(result, expected)

    @patch('time.sleep', return_value=None)
    @patch.object(DynamoDBDataLoader, 'dynamodb')
    def test_batch_write_items(self, mock_dynamodb: MagicMock, mock_sleep: MagicMock) -> None:
        """
        Test the batch_write_items method to ensure it correctly batches and writes items to DynamoDB, and properly handles sleep.

        Args:
            mock_dynamodb (MagicMock): Mock of the DynamoDB table interface.
            mock_sleep (MagicMock): Mock of the time.sleep function to control its behavior during the test.

        Asserts:
            Verifies that items are batched and written correctly and that sleep is invoked correctly between batches.
        """
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        mock_batch_writer = MagicMock()
        mock_table.batch_writer.return_value.__enter__.return_value = mock_batch_writer

        data = [{'datetime': '2021-01-01T00:00:00Z', 'site': 'TestSite'}] * 30  # 30 items to ensure two batches
        self.loader.batch_write_items('TestTable', data, sleep_time=1.0)

        self.assertEqual(mock_batch_writer.put_item.call_count, 30)
        self.assertEqual(mock_sleep.call_count, 1)  # sleep should be called once as there are two batches

if __name__ == '__main__':
    unittest.main()