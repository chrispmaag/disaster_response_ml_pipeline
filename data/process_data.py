# Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load datasets and merge into single dataframe."""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets
    df = pd.merge(messages, categories, how='inner', on='id')

    return df


def clean_data(df):
    """Create columns for each category, clean up data, then concat.

    After splitting categories into columns, we rename the columns with
    the beginning of each column. We also clean up the values using only the
    last character and convert any 2s to 1s.

    Args:
        df: Dataframe to be processed.

    Returns:
        df: Cleaned Dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories[:1]

    # use this row to extract a list of new column names for categories
    # take beginning up to 2nd to last character of each string with slicing
    category_colnames = [name[:-2] for name in row.values[0]]

    # rename the columns of 'categories'
    categories.columns = category_colnames

    # Convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [x[-1] for x in categories[column]]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

        # replace value of 2 with a 1
        categories[column] = categories[column].replace(2, 1)

    # Replace 'categories' column in df with new category columns
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filepath):
    """Write dataframe to SQL table.

    Args:
        df: Dataframe to write.
        database_filepath: Where database is located.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('cleanData', con=engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
