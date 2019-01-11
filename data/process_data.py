import argparse
import pandas as pd
parser = argparse.ArgumentParser(description = 'Getting filepaths and database  name')
parser.add_argument('fpath_messages', help = 'Filepath to disaster messages file')
parser.add_argument('fpath_categories', help = 'Filepath to categories file')
parser.add_argument('database_name', help = 'Database name')

args = parser.parse_args()

message_filepath = args.fpath_messages
categories_filepath = args.fpath_categories
dbase_filepath = args.database_name

def load_clean_data(message_filepath, categories_filepath):
    '''Load and clean the data
    
    Args:
        message_filepath (str) : path to messages file
        categories_filepath (str): path to categories file
        
    Returns:
        clean_df : dataframe with columns id, message, original, genre
        and categories
    '''
    
    # load data from filepath
    messages = pd.read_csv(message_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge messages and categories dataframes
    df = pd.merge(left = messages, right = categories, how = 'left', on = 'id')
    
    #split categories dataframe into different category columns
    categories = categories['categories'].str.split(';', expand = True)
    
    #extracting category column names
    row = categories.iloc[0, :]
    func = lambda x: x[:-2]
    category_column_names = row.apply(func)
    
    #getting the last number from values and converting it to type 'int'
    get_last_char = lambda x: x[-1:]
    for col in categories.columns:
        categories[col] = categories[col].apply(get_last_char)
        categories[col] = categories[col].astype('int')
      
    #assign category_column_names as the column names of 'categories'
    categories.columns = category_column_names
    
    #dropping 'categories' column from df
    df.drop('categories', axis = 1, inplace = True)
    
    #concatenating df and categories dataframe
    clean_df = pd.concat([df, categories], axis = 1)
    
    #drop duplicates
    clean_df.drop_duplicates(inplace = True)
    
    return clean_df

def save_data(df, database_name):
    '''Save the dataframe df into sqlite database 
    The data is saved with a file name 'clean_data' in the
    database    
    
    Args:
        df : clean data to be saved
        database_name (str): name of the database
        
    Returns:
        None
    '''
    
    from sqlalchemy import create_engine
    
    dbase_path = 'sqlite:///' + database_name
    
    engine = create_engine(dbase_path)
    df.to_sql('clean', engine, index=False, if_exists = 'replace')
   
#----------------------------------------- main ----------------------------
def main():
    
    print('Loading data... \n   MESSAGES: {}    \n   CATEGORIES: {}'.format(message_filepath, categories_filepath))
    clean_df = load_clean_data(message_filepath, categories_filepath)
    print('Cleaning data...')
    #print('Data:------------------------------------\n')
    #print(clean_df.head())
    print('Saving the data... \n   DATABASE: {}'.format(dbase_filepath))
    save_data(clean_df, dbase_filepath)
    print('Cleaned data saved to database!')
   
    
if __name__ == '__main__':
    
    main()