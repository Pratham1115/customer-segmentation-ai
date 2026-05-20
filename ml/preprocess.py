def clean_data(df):

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove empty values
    df = df.dropna()

    return df