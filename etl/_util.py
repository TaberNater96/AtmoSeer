import boto3

def delete_all_items(table_name: str) -> None:
    """
    Delete all items from a specified DynamoDB table. This is a tool in case items need to be purged from the database.

    Args:
        table_name (str): The name of the DynamoDB table.

    Raises:
        ClientError: If the boto3 API calls fail.
    """
    
    dynamodb = boto3.resource('dynamodb', region_name='us-west-2')
    table = dynamodb.Table(table_name)
    
    # Using ExpressionAttributeNames to avoid conflicts with reserved keywords
    scan = table.scan(
        # The '#dt' is a placeholder for 'datetime', a duplicate name conflict will raise otherwise
        ProjectionExpression='#dt, site',
        ExpressionAttributeNames={'#dt': 'datetime'} 
    )
    
    with table.batch_writer() as batch:
        for each in scan['Items']:
            batch.delete_item(
                Key={
                    'datetime': each['datetime'],
                    'site': each['site']
                }
            )

# Run script to delete all items in the 'DataNOAA' table
if __name__ == '__main__':
    delete_all_items('DataNOAA') # replace with name of gas table