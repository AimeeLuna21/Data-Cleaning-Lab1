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

def prep_college_pipeline(path: str, random_state=42):
    df = load_csv(path)

    # ---- types / cleaning unique to this dataset ----
    for c in ["state", "level", "control", "basic"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # hbcu/flagship encoded as missing => convert to 0/1
    for col in ["hbcu", "flagship"]:
        if col in df.columns:
            df[col] = df[col].notna().astype(int)

    # clean counted_pct (example from your work)
    if "counted_pct" in df.columns:
        tmp = df["counted_pct"].astype(str).str.split("|", expand=True)[0]
        df["counted_pct"] = pd.to_numeric(tmp, errors="coerce") / 100.0

    # drop columns (fixing your "" bug by filtering)
    drop_cols = [
        "index","unitid","site","vsa_year",
        "vsa_grad_after4_first","vsa_grad_elsewhere_after4_first",
        "vsa_enroll_after4_first","vsa_enroll_elsewhere_after4_first",
        "vsa_grad_after6_first","vsa_grad_elsewhere_after6_first",
        "vsa_enroll_after6_first","vsa_enroll_elsewhere_after6_first",
        "vsa_grad_after4_transfer","vsa_grad_elsewhere_after4_transfer",
        "vsa_enroll_after4_transfer","vsa_enroll_elsewhere_after4_transfer",
        "vsa_grad_after6_transfer","vsa_grad_elsewhere_after6_transfer",
        "vsa_enroll_after6_transfer","vsa_enroll_elsewhere_after6_transfer",
        "med_sat_value","med_sat_percentile",
        "chronname","city","nicknames","long_x","lat_y"
    ]
    df = drop_columns(df, drop_cols)

    # collapse basic categories (your choice: top 5 + Other)
    if "basic" in df.columns:
        top_basic = df["basic"].value_counts().nlargest(5).index
        df["basic"] = df["basic"].apply(lambda x: x if x in top_basic else "Other").astype("category")

    # ---- target unique to this dataset ----
    # high graduation rate = top quartile of grad_150_value
    df = make_binary_target_from_threshold(df, source_col="grad_150_value", q=0.75, target_name="high_grad_rate")

    # drop raw outcome column to avoid leakage
    df = drop_columns(df, ["grad_150_value"])

    # ---- encode categoricals ----
    cat_cols = list(df.select_dtypes(include=["category"]).columns)
    df = one_hot_encode(df, cat_cols=cat_cols, drop_first=True)

    # ---- split FIRST (avoid leakage) ----
    train, tune, test = split_train_tune_test(df, target_col="high_grad_rate", random_state=random_state)

    # ---- scale numeric using train-fitted scaler ----
    num_cols = [c for c in train.columns if c != "high_grad_rate"]
    scaler = MinMaxScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    tune[num_cols]  = scaler.transform(tune[num_cols])
    test[num_cols]  = scaler.transform(test[num_cols])

    return train, tune, test

# %%
from sklearn.preprocessing import StandardScaler

def prep_job_pipeline(path: str, random_state=42):
    df = load_csv(path)

    # categorical casting specific to this dataset
    cat_cols = ["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation","status"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # drop ID
    df = drop_columns(df, ["sl_no"])

    # target: placed vs not placed
    df = make_binary_target_from_label(df, source_col="status", positive_label="Placed", target_name="placed_f")

    # drop leakage columns
    df = drop_columns(df, ["salary", "status"])

    # one-hot encode categoricals
    cat_cols = list(df.select_dtypes(include=["category"]).columns)
    df = one_hot_encode(df, cat_cols=cat_cols, drop_first=True)

    # split first
    train, tune, test = split_train_tune_test(df, target_col="placed_f", random_state=random_state)

    # standardize continuous variables using train-fitted scaler
    # (you can keep this list explicit like you had)
    num_cols = [c for c in ["ssc_p","hsc_p","degree_p","etest_p","mba_p"] if c in train.columns]
    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    tune[num_cols]  = scaler.transform(tune[num_cols])
    test[num_cols]  = scaler.transform(test[num_cols])

    return train, tune, test
# %%
