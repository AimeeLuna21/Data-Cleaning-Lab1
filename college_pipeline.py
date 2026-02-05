# %% [markdown]
## STEP 1 — College Completion Dataset

#### Dataset Understanding

#### Each row in the dataset represents one postsecundary institution in the United States. The dataset includes:
    #* Insitutional characteristrics 
    #* Student composition and & access measures 
    #* Academic preparedness
    #* Financial measures 
    #* Faculty & teaching resources 
    #* Outcome variables 
    #* This supports **institution-performance analysis** not individual student prediction

#### Problem Brainstorming 
    #* Why graduation rates vary across institutions
    #* How financial resources and academic preparedness relate to completion
    #* Whether institutional type (public/private, Carnegie classification, HBCU status) is associated with student success
    #* Which institutional factors are linked to higher graduation efficiency.
    #* And because the dataset combines information from many different institutions, it likely requires data preparation. Institutions vary widely in size and reporting practices, so the data needs to be examined for things like missing values, differences in how variables are recorded, and whether certain columns represent raw counts rather than rates. 

#### Research Question
#1. How are institutional financial resources and spending efficiency related to graduation outcomes across colleges?

# %% [markdown]
## Step 1-Job_Placement Dataset
##### The dataset contains **student-level data from a single college** in India and focuses on academic performance and job placement outcomes. 

###### The dataset includes:
    #* secondary school
    #* higher secondary
    #* undergraduate degree 
    #* employability test scores
    #* specialization
    #* prior work experience
    #* final placement status
    #* salary
###### Problem brainstorming: 
    #* How academic performance at different education levels relates to job placement outcomes
    #* How prior work experience affects placement likelihood and salary outcomes
    #* Whether gender is associated with placement outcomes within this single institution

###### Research Question
#1. How are academic performance and employability test scores related to job placement outcomes for students at this college?

# %%
# Imports - Libraries needed for data manipulation and ML preprocessing
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
# Make sure to install sklearn in your terminal first!
# Use: pip install scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling
from io import StringIO  # For reading string data as file
#import requests  # For HTTP requests to download data


# %%
## Step 2- College Completion Dataset
###### The independent Business Metric for this problem is student employability strength, measured through academic performance (SSC, HSC, degree, and MBA percentages), employability test scores, and prior work experience. These metrics represent factors that institutions and recruiters can influence or evaluate when assessing a student’s likelihood of being placed.
###### 1. How are institutional financial resources and spending efficiency related to graduation outcomes across colleges?

# Load the college completion dataset
c_comp = pd.read_csv("cc_institution_details.csv")
## Each row represents one postsecondary institution in the United States.
# This is institution-level data, not individual student-level data.

# Let's check the structure of the dataset 
c_comp.info()
#The dataset contains 3,798 postsecondary institutions.
# There are 63 variables
# Dataset observations:
# - 3,798 institutions with 63 variables
# - Mix of numeric and categorical features
# - Indicator-style variables (hbcu, flagship) use missingness to encode "No"
# - Some outcome-related variables have missing cohort data
# - Raw counts and rate-based measures are mixed# %%

# %%
# convert categorical columns to category dtype
cat_cols = [
    "state",
    "level",
    "control",
    "basic"
]

# astype() changes the data type of specified columns
# 'category' is a special pandas dtype for categorical variables
c_comp[cat_cols] = c_comp[cat_cols].astype("category")

# HBCU & Flagship are encoded as missing values for "No" and non-missing for "Yes" so making them be 1 or 0 helps keep the info
# notna() checks whether a value is not missing and return TRUE-yes HBCU and no-missing; same for flagship
# astype("int") converts TRUE to 1 and FALSE to 0
c_comp["hbcu"] = c_comp["hbcu"].notna().astype("int")
c_comp["flagship"] = c_comp["flagship"].notna().astype("int")

# check if conversion worked
c_comp.dtypes
# %%
# Drop columns that are not needed for analysis for our question and that have a lot of missing data/ not relevant to institutional characteristics, financial resources, or graduation outcomes. 
drop_cols = [
    "index",
    "unitid",
    "site",
    "vsa_year",
    "vsa_grad_after4_first",
    "vsa_grad_elsewhere_after4_first",""
    "vsa_enroll_after4_first",
    "vsa_enroll_elsewhere_after4_first",
    "vsa_grad_after6_first",
    "vsa_grad_elsewhere_after6_first",
    "vsa_enroll_after6_first",
    "vsa_enroll_elsewhere_after6_first",
    "vsa_grad_after4_transfer",
    "vsa_grad_elsewhere_after4_transfer",
    "vsa_enroll_after4_transfer",
    "vsa_enroll_elsewhere_after4_transfer",
    "vsa_grad_after6_transfer",
    "vsa_grad_elsewhere_after6_transfer", 
    "vsa_enroll_after6_transfer",
    "vsa_enroll_elsewhere_after6_transfer",
    "med_sat_value",
    "med_sat_percentile",
    "chronname",
    "city",
    "state",
    "nicknames",
    "long_x",
    "lat_y",
]
# Drop the specified columns from the DataFrame
c_comp = c_comp.drop(columns=drop_cols)

c_comp.info()
# %%
c_comp["counted_pct"].head(50)


# %%

# The "counted_pct" column contains values like "85|the year" so split on | and keep only the percentage part.
c_comp["counted_pct_clean"] = (
    c_comp["counted_pct"]
    .str.split("|", expand=True)[0]
)

# Convert the cleaned "counted_pct_clean" column to numeric
c_comp["counted_pct_clean"] = pd.to_numeric(
    c_comp["counted_pct_clean"],
    errors="coerce"
)

# Now have a numeric "counted_pct_clean" column, but the values are in percentage form so now values are between 0 and 1
c_comp["counted_pct_clean"] = c_comp["counted_pct_clean"] / 100

# Drop the original "counted_pct" column since we now have a cleaned numeric version
c_comp = c_comp.drop(columns=["counted_pct"])

# Rename the cleaned column to "counted_pct" to maintain the original column name for analysis
c_comp = c_comp.rename(
    columns={"counted_pct_clean": "counted_pct"}
)


# %%
# check how many categories there are
c_comp["basic"].value_counts()
#PROBLEM there are 30+ categories
# sparse data, overfitting risk, so need to do grouping 
# %%
# Keep top 5 most common Carnegie basic categories to reduce sparsity
top_basic = c_comp["basic"].value_counts().nlargest(5).index

c_comp["basic"] = (
    c_comp["basic"]
    .apply(lambda x: x if x in top_basic else "Other")
    .astype("category")
)

# Verify result
c_comp["basic"].value_counts()
# %%
#check how many categories there are
c_comp["level"].value_counts()
#only 2 categories: 4-year and 2-year, so keep them
# %%
# check how many categories there are
c_comp["control"].value_counts()
#only 3 categories: public, private nonprofit, private for-profit
# so keep them 
# %%
# seperate numeric vs categorical columns

# Identify numeric and categorical columns
num_cols = c_comp.select_dtypes(include=["int64", "float64"]).columns
cat_cols = c_comp.select_dtypes(include=["category"]).columns

print("Numeric columns:", list(num_cols))
print("Categorical columns:", list(cat_cols))

# %%
#One-Hot Encode categorical variables
cat_cols = list(c_comp.select_dtypes("category"))

c_comp_encoded = pd.get_dummies(
    c_comp,
    columns=cat_cols,
    drop_first=True  # avoids dummy variable trap
)

c_comp_encoded.info()


# %%
#nornalize min-max scaling for numeric columns
from sklearn.preprocessing import MinMaxScaler

num_cols = list(c_comp_encoded.select_dtypes("number"))

scaler = MinMaxScaler()
c_comp_encoded[num_cols] = scaler.fit_transform(c_comp_encoded[num_cols])

# %%
#create target variable (binary)
#The independent Business Metric is the institutional graduation rate at 150% of normal time (grad_150_value), which is transformed into a binary target variable for modeling.
# Check distribution
c_comp_encoded["grad_150_value"].describe()



# %%
# Use 75th percentile as threshold for high graduation rate
threshold = c_comp_encoded["grad_150_value"].quantile(0.75)

c_comp_encoded["high_grad_rate"] = (
    c_comp_encoded["grad_150_value"] > threshold
).astype(int)
# %%
# calculate prevalence of high graduation rate
prevalence = c_comp_encoded["high_grad_rate"].mean()
print(f"Prevalence (baseline): {prevalence:.2%}")

#This represents the proportion of institutions with graduation rates in the top quartile. Any predictive model should outperform this baseline.

# %%
#drop unneeded columns
#cannot keep raw graduation rate since it's used to create target
c_comp_final = c_comp_encoded.drop(columns=["grad_150_value"])

# %%
#train/tune/test split
from sklearn.model_selection import train_test_split

train, temp = train_test_split(
    c_comp_final,
    train_size=0.6,
    stratify=c_comp_final["high_grad_rate"],
    random_state=42
)

tune, test = train_test_split(
    temp,
    train_size=0.5,
    stratify=temp["high_grad_rate"],
    random_state=42
)

print(train.shape, tune.shape, test.shape)

# %%
#check prevalence in each split
print(train["high_grad_rate"].mean())
print(tune["high_grad_rate"].mean())
print(test["high_grad_rate"].mean())


# %% [markdown]
## Step 3- Insticts about Data

###### My initial instinct is that this dataset is well suited to address the problem of understanding factors associated with higher institutional graduation rates. The outcome variable, graduation rate at 150% of normal time (grad_150_value), is widely used performance metric in higher education, and the dataset includes a set of potential predictors related to institutional resources, student composition, and institutional type (e.g., financial aid, endowment size, Pell grant share, enrollment, and Carnegie classification). This suggests the data can support modeling differences in graduation outcomes across institutions.

###### That said, there are several areas of concern. First, many variables—particularly endowment measures, SAT scores, and VSA-related columns—have missingness, which may limit their usefulness or require careful exclusion. Second, institutional characteristics such as Carnegie classification and control type are highly correlated with resource variables, raising the risk of predictors being highly related. These issues suggest the need for smart variable selection, normalization, and interpretation of model results.

# %% [markdown]
### Step 2: Job Placement Dataset
####### 1. How are academic performance and employability test scores related to job placement outcomes for students at this college?
####### Independent Business Metric (Job Placement Dataset): An appropriate independent business metric for this problem is student academic performance prior to graduation, measured through standardized exam scores and grade percentages (SSC, HSC, degree percentage, MBA percentage, and employability test score). From a business perspective, these metrics represent student quality and preparedness, which institutions aim to improve through curriculum design, academic support, and admissions standards. These inputs are controllable or influenceable by the institution and are expected to be associated with the likelihood of successful job placement.
# %%

# Load the job placement dataset
job = pd.read_csv("Placement_Data_Full_Class.csv")
# Let's check the structure of the dataset 
c_comp.info()

# Each row represents an individual student from a single college.
# The dataset includes academic performance, background information,
# employability test scores, and final job placement outcomes.
# %%
# Convert categorical string variables to 'category' dtype
# This improves memory efficiency and ensures correct handling
# in statistical modeling and machine learning

cat_cols = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "workex",
    "specialisation",
    "status"
]

job[cat_cols] = job[cat_cols].astype("category")

# Verify the changes
job.dtypes
# %%
# Drop student ID column
# 'sl_no' is a unique identifier and provides no predictive value
job = job.drop(columns=["sl_no"])

# %%
# Create binary target variable for job placement
# placed_f represents whether a student successfully secured a job
# 1 = Placed
# 0 = Not Placed

job["placed_f"] = (job["status"] == "Placed").astype(int)

# %%
# Drop post-outcome variable
# Salary is only observed after placement and would cause data leakage
job = job.drop(columns=["salary", "status"])

# %%
# Review category sizes to determine whether factor level collapsing is needed
# All categorical variables have a small number of well-represented levels.
# While some categories (e.g., Arts in hsc_s and Others in degree_t) are smaller,
# they still contain meaningful information and are retained.
# Therefore, no factor level collapsing is performed.

for col in job.select_dtypes("category").columns:
    print("\n", col)
    print(job[col].value_counts())

# %%
# Calculate prevalence (baseline probability of placement)
prevalence = job["placed_f"].mean()
print(f"Baseline / Prevalence: {prevalence:.2%}")

# %%
num_cols = [
    "ssc_p",
    "hsc_p",
    "degree_p",
    "etest_p",
    "mba_p"
]

# %%
# %%
# Identify categorical variables to be one-hot encoded
# These are all variables with type 'str'
cat_cols = [
    'gender',
    'ssc_b',
    'hsc_b',
    'hsc_s',
    'degree_t',
    'workex',
    'specialisation'
]

# %%
# %%
# Verify that all categorical columns exist in the dataframe
set(cat_cols).issubset(job.columns)


# %%
# %%
# One-hot encode categorical variables
# drop_first=True is used to avoid multicollinearity
job_encoded = pd.get_dummies(
    job,
    columns=cat_cols,
    drop_first=True
)


# %%
# %%
# Review structure after encoding
job_encoded.info()
## After one-hot encoding, all categorical variables have been converted
# into binary indicator variables. The dataset now contains only numeric
# and boolean features, making it suitable for modeling.

# %%
# %%
from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Normalize continuous variables
job_encoded[num_cols] = scaler.fit_transform(job_encoded[num_cols])
# Continuous variables are standardized to ensure that features
# measured on different scales contribute equally to the model.

# %%
job_encoded[num_cols].describe()
#verify it worked
# %%
# %%
# Calculate prevalence of the target variable
job_encoded['placed_f'].value_counts(normalize=True)

# %%
from sklearn.model_selection import train_test_split

# Separate predictors and target
X = job_encoded.drop(columns="placed_f")
y = job_encoded["placed_f"]

# First split: Training set (60%) and temporary set (40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

# Second split: Tuning set (20%) and Test set (20%)
X_tune, X_test, y_tune, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Confirm partition sizes
print("Train:", X_train.shape)
print("Tune:", X_tune.shape)
print("Test:", X_test.shape)

# %% [markdown]
## Step 3- Insticts about Data

####### My instinct is that this dataset can definitely address the problem of understanding what factors are associated with job placement after graduation. The target variable (placement status) is clear and directly tied to the question, and the dataset includes reasonable predictors like academic performance, employability test scores, work experience, and specialization. These all make intuitive sense as things that would affect whether a student gets placed.

####### That said, there are a few things that worry me. The dataset is fairly small (only 215 students), which could limit how complex or stable a model can be. Some predictors are also closely related to each other (for example, multiple academic percentage variables), which could lead to redundancy or multicollinearity. Finally, salary data had to be dropped because it’s only observed after placement, which means we lose some potentially useful information. 
# %%
