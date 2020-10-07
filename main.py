import random
from time import clock
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def initial_data_cleaning():
    user_interactions_df = pd.read_csv('users_interactions.csv')
    shared_articles_df = pd.read_csv('shared_articles.csv')

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
    print("== Remove NaN's ==")
    print(cols)

    virality_df = virality_df[cols]

    # Add word count of 'text' column and drop it
    virality_df['textWordCount'] = virality_df['text'].str.split().str.len()
    virality_df = virality_df.drop('text', 1)
    print(virality_df)

    # Export for future loading
    virality_df.to_csv('virality.csv', index=False)

    # drop cols irrelevant to prediction for now
    virality_df = virality_df.drop('title', 1)
    virality_df = virality_df.drop('url', 1)

    # Histogram of features
    virality_df.hist(figsize=(10, 10))
    plt.savefig('virality_hist.png')

    # Correlation heat mapping
    create_corr_heat_map(virality_df, 'virality_corr')

    # get category cols for 1 hot encoding
    cat_cols = get_cols_w_no_nans(virality_df, 'no_num')  # cols dropped, so update
    virality_df = one_hot_encode(virality_df, cat_cols)

    # Correlation heat map w/ 1 hot encode
    create_corr_heat_map(virality_df, 'virality_corr_1he')

    # drop cols with poor total correlation with user events or virality
    virality_df = virality_df.drop('contentId', 1)
    virality_df = virality_df.drop('authorSessionId', 1)

    # Export for future loading
    virality_df.to_csv('virality_1he.csv', index=False)


def get_cols_w_no_nans(df, col_type='all'):
    """
    :param df: The dataframe to process
    :param col_type:
        num : to only get numerical columns with no nans
        no_num : to only get non-numerical columns with no nans
        all : to get any columns with no nans
    :return: row of column headers with specified data ype
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


def one_hot_encode(df, col_names):
    """
    One hot encode pandas data frame with category columns only.
    :param df: pd.df to modify
    :param col_names: row of category column headers
    :return: new pd.df with 1-hot encoding
    """
    for col in col_names:
        if df[col].dtype == np.dtype('object'):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

            # drop the encoded column
            df.drop([col], axis=1, inplace=True)
    return df


def create_corr_heat_map(df, title="corr_heat_map"):
    viral_corr = df.corr()
    fig = plt.figure(figsize=(20, 20))
    sb.heatmap(viral_corr, vmax=.8, square=True)
    plt.savefig(title + '.png')


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=None, train_sizes=np.linspace(.1, 1., 10)):
    """
    Function retrieved from:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    #sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#f92672")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#007fff")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="#f92672",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="#007fff",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(title + ".png")
    plt.close()


def plot_fit_times(estimator, title, x, y):
    out = defaultdict(dict)
    length_x = len(x)
    split_floats = np.linspace(.1, .9, 9)
    for split_float in split_floats:
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split_float)
        start_time = clock()
        clf = estimator
        clf.fit(x_train, y_train)
        out['train'][split_float] = clock() - start_time
        start_time = clock()
        clf.predict(x_test)
        out['test'][split_float] = clock() - start_time
    out = pd.DataFrame(out)
    print(out)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Time (s)")
    plt.grid()
    plt.plot(split_floats * length_x, out['test'], 'o-', color="#c1ffc1",
             label="Test set")
    plt.plot(split_floats * length_x, out['train'], 'o-', color="#50d3dc",
             label="Train set")
    plt.legend(loc="best")
    plt.savefig(title + ".png")
    plt.close()


def scale_features(input_df):
    scaler = StandardScaler()
    scaler.fit(input_df)
    scaled_features = scaler.transform(input_df)
    return pd.DataFrame(scaled_features)


def main():
    # Data Cleaning: run the first time
    initial_data_cleaning()

    virality_df = pd.read_csv('virality_1he.csv')
    target = virality_df['Virality']
    virality_df = virality_df.drop('Virality', 1)
    scaled_virality = scale_features(virality_df)

    # Random Forest Regressor
    train_X, test_X, train_y, test_y = train_test_split(scaled_virality, target,
                                                        test_size=0.2,
                                                        random_state=8)

    model = RandomForestRegressor()
    model.fit(train_X, train_y)

    # Get the mean absolute error on the validation data
    predicted_prices = model.predict(test_X)
    MAE = mean_absolute_error(test_y, predicted_prices)
    print('Random forest validation MAE = ', MAE)

    model = MLPRegressor()
    model.fit(train_X, train_y)

    # Get the mean absolute error on the validation data
    predicted_prices = model.predict(test_X)
    MAE = mean_absolute_error(test_y, predicted_prices)
    print('Neural Network validation MAE = ', MAE)

    # cross validation with plotting
    plot_fit_times(RandomForestRegressor(), "Rand Forest Regressor Fit Time",
                   scaled_virality, target)
    plot_learning_curve(RandomForestRegressor(),
                        "Rand Forest Regressor", scaled_virality, target)

    plot_fit_times(MLPRegressor(activation='relu', hidden_layer_sizes=10),
                   "Neural Network Fit Time", scaled_virality, target)
    plot_learning_curve(MLPRegressor(activation='relu', hidden_layer_sizes=10),
                        "Neural Network", scaled_virality, target)


if __name__ == '__main__':
    main()
