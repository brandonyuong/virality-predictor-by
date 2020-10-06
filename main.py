import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb


def initial_data_cleaning():
    user_interactions_df = pd.read_csv('users_interactions.csv')
    shared_articles_df = pd.read_csv('shared_articles.csv')
    print(shared_articles_df)

    # group event types
    user_interactions_df['COUNTER'] = 1  # initial count = 1
    user_int_event_count = user_interactions_df.groupby(['contentId', 'eventType'])[
        'COUNTER'].sum().reset_index()  # sum function
    print("== Grouped Events ==")
    print(user_int_event_count)

    events_df = user_int_event_count.pivot_table('COUNTER', ['contentId'], 'eventType')
    events_df = events_df.fillna(0)
    print("== events_df ==")
    print(events_df)

    events_df['Virality'] = (events_df['VIEW'] + 4 * events_df['LIKE'] + 10 * events_df[
        'COMMENT CREATED'] + 25 * events_df['FOLLOW'] + 100 * events_df['BOOKMARK'])

    # Create left join of 'events' and 'shared_art' dfs
    virality_df = pd.merge(events_df, shared_articles_df, on='contentId', how='left')

    # Remove columns with missing/NaN/null values
    num_cols = get_cols_w_no_nans(virality_df)
    cat_cols = get_cols_w_no_nans(virality_df, 'no_num')

    print('Number of numerical columns with no nan values :', len(num_cols))
    print('Number of nun-numerical columns with no nan values :', len(cat_cols))

    cols = get_cols_w_no_nans(virality_df)
    print(cols)

    virality_df = virality_df[cols]

    # Add word count of 'text' column and drop it
    virality_df['textWordCount'] = virality_df['text'].str.split().str.len()
    virality_df = virality_df.drop('text', 1)
    print(virality_df)

    # Histogram of features
    virality_df.hist(figsize=(12, 10))
    plt.savefig('virality_hist.png')

    # Correlation heat mapping
    viral_corr = virality_df.corr()
    fig = plt.figure(figsize=(20, 20))
    sb.heatmap(viral_corr, vmax=.8, square=True)
    plt.savefig('virality_corr.png')

    # Export for future loading
    virality_df.to_csv('virality.csv', index=False)


def get_cols_w_no_nans(df, col_type='all'):
    """
    :param df: The dataframe to process
    :param col_type:
        num : to only get numerical columns with no nans
        no_num : to only get non-numerical columns with no nans
        all : to get any columns with no nans
    :return:
    """
    if col_type == 'num':
        predictors = df.select_dtypes(exclude=['object'])
    elif col_type == 'no_num':
        predictors = df.select_dtypes(include=['object'])
    else:
        predictors = df

    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


def main():
    # Data Cleaning: run the first time
    #initial_data_cleaning()

    virality_df = pd.read_csv('virality.csv')

    target = virality_df['Virality']

    # drop cols with poor total correlation
    virality_df = virality_df.drop('contentId', 1)
    virality_df = virality_df.drop('authorPersonalId', 1)
    virality_df = virality_df.drop('authorSessionId', 1)

    # need one hot encode before correlation

if __name__ == '__main__':
    main()
