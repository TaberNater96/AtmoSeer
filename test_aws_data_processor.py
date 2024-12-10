import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from aws_data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """
    Unit tests for the DataProcessor class which handles data fetching and processing from AWS DynamoDB.

    Attributes:
        table_name (str): Name of the DynamoDB table used for testing.
        region (str): AWS region used for testing.
        processor (DataProcessor): Instance of the DataProcessor.
    """
    
    def setUp(self) -> None:
        """
        Set up method to initialize the test environment.
        Creates an instance of DataProcessor with predefined table name and region.
        """
        self.table_name: str = 'example_table'
        self.region: str = 'us-west-2'
        self.processor: DataProcessor = DataProcessor(self.table_name, self.region)

    def test_init(self) -> None:
        """
        Test the __init__ method to ensure proper initialization of DataProcessor instance.
        """
        self.assertEqual(self.processor.table_name, 'example_table')
        self.assertEqual(self.processor.dynamodb.meta.region_name, 'us-west-2')

    @patch('boto3.resource')
    def test_fetch_data_by_key(self, mock_dynamodb) -> None:
        """
        Test the fetch_data_by_key method to ensure it correctly queries DynamoDB and handles pagination.
        Also tests the method's error handling.
        
        Args:
            mock_dynamodb (MagicMock): Mock object for the boto3 DynamoDB resource.
        """
        mock_table = mock_dynamodb.return_value.Table.return_value
        mock_table.query.return_value = {
            'Items': [{'id': 1, 'data': 'test'}],
            'LastEvaluatedKey': 'key123'
        }

        # Mocking continuation of data fetching
        second_page = {
            'Items': [{'id': 2, 'data': 'test2'}]
        }
        mock_table.query.side_effect = [mock_table.query.return_value, second_page]

        result = self.processor.fetch_data_by_key('2021-01-01T00:00:00Z', '2021-01-02T00:00:00Z', 'SiteA')
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)  # check if both pages of items were fetched

        # Testing exception handling
        mock_table.query.side_effect = Exception('DynamoDB query failed')
        result = self.processor.fetch_data_by_key('2021-01-01T00:00:00Z', '2021-01-02T00:00:00Z', 'SiteA')
        self.assertTrue(result.empty)

    @patch('boto3.resource')
    def test_fetch_all_data(self, mock_dynamodb) -> None:
        """
        Test the fetch_all_data method to ensure it correctly handles the pagination of DynamoDB scan.
        
        Args:
            mock_dynamodb (MagicMock): Mock object for the boto3 DynamoDB resource.
        """
        mock_paginator = mock_dynamodb.return_value.meta.client.get_paginator.return_value
        mock_paginator.paginate.return_value = [
            {'Items': [{'id': 1, 'data': 'all1'}], 'NextToken': 'token123'},
            {'Items': [{'id': 2, 'data': 'all2'}]}
        ]

        result = self.processor.fetch_all_data()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

        # Testing error handling
        mock_paginator.paginate.side_effect = Exception('DynamoDB scan failed')
        result = self.processor.fetch_all_data()
        self.assertTrue(result.empty)

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='SELECT * FROM data_table')
    def test_load_sql_query(self, mock_file) -> None:
        """
        Test the load_sql_query method to ensure it correctly loads SQL queries from a file.
        
        Args:
            mock_file (unittest.mock.mock_open): Mock object for file operations.
        """
        query = self.processor.load_sql_query()
        self.assertEqual(query, 'SELECT * FROM data_table')

    @patch('aws_data_processor.DataProcessor.load_sql_query', return_value='SELECT * FROM data_table')
    @patch('pandasql.sqldf', return_value=pd.DataFrame({'id': [1], 'data': ['processed']}))
    def test_process_data(self, mock_sqldf, mock_load_sql) -> None:
        """
        Test the process_data method to ensure it correctly processes data using a SQL query.
        
        Args:
            mock_sqldf (MagicMock): Mock object for the pandasql.sqldf function.
            mock_load_sql (MagicMock): Mock object for the load_sql_query method.
        """
        raw_data = pd.DataFrame({'id': [1], 'data': ['raw']})
        processed_data = self.processor.process_data(raw_data)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(processed_data.iloc[0]['data'], 'processed')

if __name__ == '__main__':
    unittest.main()