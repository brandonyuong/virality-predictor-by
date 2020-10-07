import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
import scikit-learn as sk


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


def neural_net(train_df):
    # Initialize NN
    nn_model = tf.keras.Sequential()

    # Input Layer
    nn_model.add(tf.keras.layers.Dense(128, kernel_initializer='normal',
                                       input_dim=train_df.shape[1],
                                       activation='relu'))
    # Hidden Layers
    nn_model.add(
        tf.keras.layers.Dense(256, kernel_initializer='normal', activation='relu'))
    nn_model.add(
        tf.keras.layers.Dense(256, kernel_initializer='normal', activation='relu'))
    nn_model.add(
        tf.keras.layers.Dense(256, kernel_initializer='normal', activation='relu'))

    # Output Layer
    nn_model.add(
        tf.keras.layers.Dense(1, kernel_initializer='normal', activation='linear'))

    # Compile the network
    nn_model.compile(loss='mean_absolute_error', optimizer='adam',
                     metrics=['mean_absolute_error'])

    #nn_model.summary()

    return nn_model


def main():
    # Data Cleaning: run the first time
    # initial_data_cleaning()

    virality_df = pd.read_csv('virality_1he.csv')
    # train = virality_df[:2400]
    # test = virality_df[2400:]
    # train_target = train['Virality'].copy()
    # test_target = test['Virality'].copy()
    # train = train.drop('Virality', 1)
    # test = test.drop('Virality', 1)
    #
    # nn = neural_net(train)
    #
    # checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_name, monitor='val_loss',
    #                                                 verbose=1,
    #                                                 save_best_only=True, mode='auto')
    # callbacks_list = [checkpoint]
    #
    # nn.fit(train, train_target, epochs=100, batch_size=32, validation_split=0.2,
    #        callbacks=callbacks_list)
    #
    # # Load wights file of the best model :
    # # weights_file = 'Weights-478--18738.19831.hdf5'  # choose the best checkpoint
    # # nn.load_weights(weights_file)  # load it
    # nn.compile(loss='mean_absolute_error', optimizer='adam',
    #            metrics=['mean_absolute_error'])
    #
    # predictions = nn.predict(test)
    # print(predictions)
    # sum_squared_errors = pd.DataFrame.sum((predictions ** 2 + test_target[:, 1] ** 2), axis=1)
    # print(sum_squared_errors)

    # Random Forest Regressor
    train_X, val_X, train_y, val_y = train_test_split(train, target, test_size=0.25,
                                                      random_state=14)

    model = RandomForestRegressor()
    model.fit(train_X, train_y)

    # Get the mean absolute error on the validation data
    predicted_prices = model.predict(val_X)
    MAE = mean_absolute_error(val_y, predicted_prices)
    print('Random forest validation MAE = ', MAE)

    # Xgboost Regressor
    XGBModel = XGBRegressor()
    XGBModel.fit(train_X, train_y, verbose=False)

    # Get the mean absolute error on the validation data :
    XGBpredictions = XGBModel.predict(val_X)
    MAE = mean_absolute_error(val_y, XGBpredictions)
    print('XGBoost validation MAE = ', MAE)


if __name__ == '__main__':
    main()
