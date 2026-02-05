# %%
def load_data(path):
    """
    Load a dataset from a CSV file.
    """
    df = pd.read_csv(path)
    return df

# %%
def create_binary_target(df, source_col, positive_label, target_name):
    """
    Create a binary target variable.
    
    source_col: original column used to create target
    positive_label: value that maps to 1
    target_name: name of new target column
    """
    df[target_name] = (df[source_col] == positive_label).astype(int)
    return df



# %%
def drop_columns(df, cols_to_drop):
    """
    Drop columns that are not needed for analysis
    (IDs, post-outcome variables, etc.).
    """
    df = df.drop(columns=cols_to_drop)
    return df

# %%
from sklearn.preprocessing import MinMaxScaler

def normalize_columns(df, num_cols):
    """
    Normalize numeric columns using Min-Max scaling.
    """
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df
 # %%
 def one_hot_encode(df, cat_cols):
    """
    One-hot encode categorical variables.
    drop_first=True avoids multicollinearity.
    """
    df_encoded = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True
    )
    return df_encoded
# %%
def calculate_prevalence(df, target_col):
    """
    Calculate baseline prevalence of the target variable.
    """
    return df[target_col].mean()

# %%
from sklearn.model_selection import train_test_split

def split_data(df, target_col, train_size=0.6, random_state=42):
    """
    Split dataset into Train, Tune, and Test sets
    using stratification on the target variable.
    """
    train, temp = train_test_split(
        df,
        train_size=train_size,
        stratify=df[target_col],
        random_state=random_state
    )
    
    tune, test = train_test_split(
        temp,
        train_size=0.5,
        stratify=temp[target_col],
        random_state=random_state
    )
    
    return train, tune, test
