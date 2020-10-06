import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import clock


def data_cleaning():
    user_interactions_df = pd.read_csv('users_interactions.csv')
    shared_articles_df = pd.read_csv('shared_articles.csv')

    # scrub out junk rows
    shared_articles_df = shared_articles_df[shared_articles_df.eventType != 'CONTENT']

    # Create left join of 'user_int' and 'shared_art' dfs
    virality_df = pd.merge(user_interactions_df, shared_articles_df, on='contentId',
                           how='left')
    virality_df.to_csv('virality.csv')


def main():
    virality_df = pd.read_csv('virality.csv')


if __name__ == '__main__':
    main()
