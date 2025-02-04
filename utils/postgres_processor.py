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
    engine = create_engine(f'postgresql://{username}:{password}@localhost:5432/gml_ghg')
    df = pd.read_sql_table(table_name, engine)
    engine.dispose()
    df['date'] = df['date'].dt.date # remove time parameter
    df.drop(columns=['gas'], inplace=True) # remove the gas column as it is only needed for merges
    
    return df