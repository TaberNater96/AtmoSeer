import pandas as pd
from pandasql import sqldf
import boto3
from botocore.exceptions import ClientError
from IPython.display import display, clear_output

class DataProcessor:
    """
    A class to handle data fetching and processing from AWS DynamoDB, using SQL queries for data transformation.
    
    Attributes:
        table_name (str): Name of the DynamoDB table to fetch data from.
        region (str): AWS region where the DynamoDB is hosted.
        dynamodb (resource): Boto3 DynamoDB resource.
        pysqldf (function): Function to execute SQL queries against a pandas DataFrame.
    """    
    def __init__(self, table_name: str, region: str = 'us-west-2', max_rcu: int = 22):
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table_name = table_name
        self.max_rcu = max_rcu
        self.pysqldf = lambda q: sqldf(q, globals())  # this lambda function allows SQL operations on pandas DataFrames

    def fetch_data_by_key(self, start_datetime: str, end_datetime: str, site: str) -> pd.DataFrame:
        """
        Fetch data from a specific DynamoDB table using specific time and location parameters.
        
        Parameters:
            start_datetime (str): Start datetime for the query range, formatted as ISO 8601 string.
            end_datetime (str): End datetime for the query range, formatted as ISO 8601 string.
            site (str): Site to query data from.
            
        Raises:
            ClientError: If the boto3 API calls fail.

        Returns:
            pd.DataFrame: DataFrame containing data fetched based on key conditions. Returns an empty DataFrame on error.
        """
        try:
            response = self.table.query(
                KeyConditionExpression="datetime BETWEEN :start and :end AND site = :site",
                ExpressionAttributeValues={
                    ":start": {"S": start_datetime},
                    ":end": {"S": end_datetime},
                    ":site": {"S": site}
                }
            )
            data = response['Items']
            display("Starting data retrieval from DynamoDB...")

            while 'LastEvaluatedKey' in response:
                response = self.table.query(
                    KeyConditionExpression="datetime BETWEEN :start and :end AND site = :site",
                    ExpressionAttributeValues={
                        ":start": {"S": start_datetime},
                        ":end": {"S": end_datetime},
                        ":site": {"S": site}
                    },
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                data.extend(response['Items'])
                
        except ClientError as e:
            print(f"An error occurred while querying DynamoDB: \n{e}")
            return pd.DataFrame()  # return an empty DataFrame or re-raise exception
        
        clear_output(wait=True) 
        display("Data retrieval completed.")
        return pd.DataFrame(data)

    def fetch_all_data(self) -> pd.DataFrame:
        """
        Fetches all data from the specified DynamoDB table while managing RCU consumption by limiting the fetch size.
        
        Raises:
            ClientError: If the boto3 API calls fail.
        
        Returns:
            pd.DataFrame: DataFrame containing all data fetched from the DynamoDB table.
        """
        try:
            paginator = self.dynamodb.meta.client.get_paginator('scan')
            operation_parameters = {
                'TableName': self.table_name,
                'PaginationConfig': {'PageSize': self.max_rcu}
            }
            
            data = []
            display("Starting full table data retrieval from DynamoDB...")
            
            for page in paginator.paginate(**operation_parameters):
                data.extend(page['Items'])
            
            clear_output(wait=True)
            display("Full table data retrieval completed.")
            return pd.DataFrame(data)
        
        except ClientError as e:
            print(f"An error occurred while scanning DynamoDB: \n{e}")
            return pd.DataFrame()  # return an empty DataFrame in case of an error

    def load_sql_query(self) -> str:
        """
        Load the SQL query from an external file to be used for data processing.
        
        Returns:
            str: A string containing the SQL query loaded from the file.
        """
        with open('query.sql', 'r') as file:
            query = file.read()
        return query

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process the data using SQL loaded from an external file, transforming it into the required structure.
        
        Parameters:
            data (pd.DataFrame): The raw data DataFrame to be processed.
        Returns:
            pd.DataFrame: Processed DataFrame with specified transformations and ordering applied.
        """
        globals()['data_table'] = data  # make data available to pandasql
        sql_query = self.load_sql_query()
        return self.pysqldf(sql_query)