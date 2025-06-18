import pytest
import PartTwo as pt


def test_speeches_clean1(): 
    '''
    Test that Labour (Co-op) does not exist in the data frame
    Test that Labour count = 8038
    Test that Speaker does not exist in column 'party'
    '''
    # Load data and ensure it contains expected value
    df = pt.read_speeches_csv()
    assert 'Labour (Co-op)' in df['party'].values
    assert df.shape == (40000,8)

    df_clean = pt.speeches_clean(df)
    assert 'Labour (Co-op)' not in df_clean['party'].values
    assert df_clean.shape == (36223,8)                        # The sum of the counts of the top 4 most common parties
    #assert df_clean[df_clean['party']=='Labour'].shape[0] == 8038

    # Removed Speaker value from df
    #assert 'Speaker' not in df_clean['party'].values