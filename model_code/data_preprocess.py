'''
Author : Dan Gawne
Date   : 2021-01-19
'''

#%%
#--------------------------------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------------------------------
import yaml
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#%%
#--------------------------------------------------------------------------------------------------
# Get File Paths
#--------------------------------------------------------------------------------------------------
with open('file_paths.yml','r') as file:
    file_paths = yaml.load(file, Loader=yaml.FullLoader)

data_dir = file_paths['data_dir'][0]

#%%
#--------------------------------------------------------------------------------------------------
# Load Data and Trim
#--------------------------------------------------------------------------------------------------
df = pd.read_csv(os.path.join(data_dir,'raw_data.csv'), delimiter=';')

drop_cols = [
    'education',
    'day_of_week',
    'contact',
    'month'
]

df.drop(columns = drop_cols, inplace = True)

#%%
#--------------------------------------------------------------------------------------------------
# Augment Data 
#--------------------------------------------------------------------------------------------------
categorical_columns = [
    'marital','job','default','housing','loan','poutcome'
]
numerical_columns = [
    'age','duration','campaign','pdays','previous',
    'cons.conf.idx', 'euribor3m', 'nr.employed'
]
target_column = 'y'

# Add numerical columns first
final_df = df[numerical_columns]

# Label encode, then one-hot encode categorical data
le = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

    enc_df = pd.DataFrame(enc.fit_transform(df[[col]]).toarray())
    col_names = {c:col+'_'+str(c) for c in enc_df.columns}
    enc_df.rename(columns = col_names, inplace = True)
    
    final_df = pd.concat([final_df,enc_df],axis = 1)

df['y'] = df['y'].map({'no':0,'yes':1}).astype(int)
final_df['y'] = df['y']

final_df.to_csv(os.path.join(data_dir,'augmented_data.csv'),index = False)

