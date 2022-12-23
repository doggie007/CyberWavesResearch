'''import pandas as pd

# read in the csv file
df = pd.read_csv('./data/Test_data.csv')

# edit the values in the last column
df.loc[df['label'] != 2, 'label'] = 1

# save the edited data to a new csv file
df.to_csv('new_train_data.csv', index=False)'''

import pandas as pd

# Load the CSV file into a pandas dataframe
df = pd.read_csv('./data/train_new.csv')

# Select only the columns with the desired headers
df = df[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'FDI', 'NDVI', 'label']]

# Save the edited data to a new CSV file
df.to_csv('Train_extra.csv', index=False)
