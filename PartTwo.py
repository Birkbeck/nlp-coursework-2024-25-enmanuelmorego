# Libraries
from pathlib import Path
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer


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
    parties_count = df_cleaned['party'].value_counts().sort_values(ascending=False)
    # # Extract the name of the 4 most common parties 
    top4_parties = parties_count.index[:4].tolist()
    # # Filter to top 4 most common parties
    df_cleaned2 = df_cleaned[df_cleaned['party'].isin(top4_parties)]

    # (a).iii Remove rows where value in 'speech_class' is not 'Speech
    df_cleaned2 = df_cleaned2[df_cleaned2['speech_class'] == 'Speech']

    return df_cleaned2

def data_pre_processing(df):
    '''
    Function to pre process the data in preparation for ML models.
        Splits into train, test sets
        Vectorises the data
    '''
    # Generate object that splits data
    splitter_obj = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 26) 
    # Split data
    for train_index, test_index in splitter_obj.split(df, df['party']):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

    # Split target in both training and testing set
    y_train, y_test = train['party'], test['party']

    # Create vectorised data for x objects
    '''Prompt indicated to use all default parameters, except for ommitting English stop words and max_featrues which needs to be set to 3000'''
    vectorizer = TfidfVectorizer(max_features = 3000, stop_words='english')
    x_train = vectorizer.fit_transform(train['speech'])
    x_test = vectorizer.transform(test['speech'])

    # Return objects
    return x_train, x_test, y_train, y_test, train['speech'], test['speech']
   

if __name__ == "__main__":
    
    '''Section A'''
    print("\n\t\t==== Section A ====\n")
    
    # Load speeches data frame
    df = read_speeches_csv()
    # Clean data frame
    df_cleaned = speeches_clean(df)
    print(f"Dimensions of cleaned speeches data frame")
    print(df_cleaned.shape)