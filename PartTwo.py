# Libraries
from pathlib import Path
import pandas as pd
import os



def read_csv(path=Path.cwd() / "texts" / "p2-texts"):
    '''
    Function to load csv files into pandas data frames

    Args:
        User can pass filepath as dictionary or key value pairs
        If no argument is given, the function defaults to reading handsard40000.csv

    Returns
        Pandas data frame
    '''
    # Extract file name
    file = os.listdir(path)[0]
    file_load = os.path.join(path, file)

    # Read data
    df = pd.read_csv(file_load)
    return df