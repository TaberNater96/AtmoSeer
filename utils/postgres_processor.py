import pandas as pd
from sqlalchemy import create_engine

def load_table(table_name, username, password):
    """
    Loads a PostgreSQL table into a pandas DataFrame.
    
    Parameters:
    - table_name (str): Name of the table to load.
    - username (str): PostgreSQL username.
    - password (str): PostgreSQL password.
    
    Returns:
    - pd.DataFrame: DataFrame containing the table data.
    """
    # Create the database engine
    engine = create_engine(f'postgresql://{username}:{password}@localhost:5432/gml_ghg')
    df = pd.read_sql_table(table_name, engine)
    engine.dispose()
    
    return df