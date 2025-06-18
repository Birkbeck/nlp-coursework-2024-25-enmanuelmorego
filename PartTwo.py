# Libraries
from pathlib import Path
import pandas as pd
import os



def read_speeches_csv(path=Path.cwd() / "texts" / "p2-texts"):
    '''
    Function to load csv files into pandas data frames

    Args:
        Function defaults to a specific location to search for the files unless otherwise specified

    Returns
        Pandas data frame
    '''
    # Extract file name
    file = os.listdir(path)[0]
    file_load = os.path.join(path, file)

    # Read data
    df = pd.read_csv(file_load)
    return df

def speeches_clean(df):
    '''
    Function that takes a data frame containing speeches, and performs custom cleaning tasks on it
    Custom cleaning tasks are:
        - Column 'party': replaces all entries 'Labour (Co-op)' with 'Labour'
        - Column 'party': removes all values where entry is 'Speaker'
        - Column 'party': only keeps the rows of the four most common parties
            Find the frequency count for each party, and keep the top 4 only
        - Column 'speech_class': removes all rows where value is NOT 'Speech'
        - Column 'speech': removes any entries where the length of the speech is less than 1000 characters
    '''
    # (a).i Clean Labour (Co-op) values
    df_cleaned = df.replace('Labour (Co-op)', 'Labour')

    # (a).ii Remove rows where 'party' == 'Speaker'
    '''Note: Remove speaker rows first, otherwise this will interfere with finding the most common parties'''
    df_cleaned = df_cleaned[df_cleaned['party'] != 'Speaker']

    # (a).ii Remove rows where the value of 'party' is not one of the 4 most common parties
    parties_count = df_cleaned['party'].value_counts().sort_values()
    # # Extract the name of the 4 most common parties 
    top4_parties = parties_count.index[:4].tolist()
    # # Filter to top 4 most common parties
    df_cleaned2 = df_cleaned[df_cleaned['party'].isin(top4_parties)]

 

    return df_cleaned2