def convert_df(df):
    """Convert dataframe to csv"""
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')