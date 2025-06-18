import PartTwo as pt


def speeches_clean1(): 
    '''
    Test that Labour (Co-op) does not exist in the data frame
    Test that Labour count = 8038
    Test that Speaker does not exist in column 'party'
    '''
    # Load data and ensure it contains expected value
    df = pt.read_speeches_csv()
    assert 'Labour (Co-op)' in df['party']

    df_clean = pt.speeches_clean(df)
    assert 'Labour (Co-op)' not in df['party']