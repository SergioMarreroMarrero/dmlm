import re

# to handle datasets
import pandas as pd
import numpy as np

# for visualization
import matplotlib.pyplot as plt

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# load the data - it is available open source and online




data = pd.read_csv('titanic.csv')


target = 'survived'
vars_num = [var
            for var in data.columns
            if data[var].dtypes != 'O' and var != target]

vars_cat = [var
           for var in data.columns
           if data[var].dtypes == 'O' and var != target]

print('Number of numerical variables: {}'.format(len(vars_num)))
print('Number of categorical variables: {}'.format(len(vars_cat)))



X_train, X_test, y_train, y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

# ============================= Extract the letter ================================ #
# ================================================================================= #
def get_first_cabin(row):
    try:
        return row[0]
    except:
        return np.nan
print(X_train['cabin'][~X_train['cabin'].isnull()].sample(5).to_list())
print(X_test['cabin'][~X_test['cabin'].isnull()].sample(5))

X_train['cabin'] = X_train['cabin'].apply(get_first_cabin)
X_test['cabin'] = X_test['cabin'].apply(get_first_cabin)

print(X_train['cabin'][~X_train['cabin'].isnull()].sample(5).to_list())
print(X_test['cabin'][~X_test['cabin'].isnull()].sample(5))
# ================================================================================= #

# ============================= Missing  ======================================= #
# ============================= Numerical Missing ================================ #
def get_mapper_median(X_train):
    mapper_median = X_train.agg('median').to_dict()
    joblib.dump(mapper_median, 'missing_numerical_mapper_median.pkl')

    return mapper_median

def create_missing_indicator(X, col):
    return np.where(X[col].isnull(), 1, 0)

def impute_na_numerical(X, col, mapper_median):
    return X[col].fillna(mapper_median[col])


vars_num_with_na = list({
    *[var for var in vars_num if X_train[var].isnull().any()],
    *[var for var in vars_num if X_test[var].isnull().any()]
})

get_mapper_median(X_train[vars_num_with_na])
mapper_median = joblib.load('missing_numerical_mapper_median.pkl')

print('========missing========')
print('========train========')
for col in X_train[vars_num_with_na].columns[X_train[vars_num_with_na].isnull().any()]:
    print(f'{col}: {X_train[col].isnull().sum()}')
print('========test========')
for col in X_test[vars_num_with_na].columns[X_test[vars_num_with_na].isnull().any()]:
    print(f'{col}: {X_test[col].isnull().sum()}')

print('')
print('############################')
print('')
# ============================================================================== #

# ======================= train ==========================
print('train')
print('===== before missing imputation ======')
print(X_train[vars_num_with_na].isnull().sum())

for col in vars_num_with_na:
    # train
    X_train[col + '_na'] = create_missing_indicator(X_train, col)
    X_train[col] = impute_na_numerical(X_train, col, mapper_median)

print('===== after missing imputation ======')
print('missing values:')
print(X_train[vars_num_with_na].isnull().sum())
print('missing indicator:')
print(X_train[[var + '_na' for var in vars_num_with_na]].sum())

# ======================= test ==========================
print('test')
print('===== before missing imputation ======')
print(X_test[vars_num_with_na].isnull().sum())

for col in vars_num_with_na:
    # train
    X_test[col + '_na'] = create_missing_indicator(X_test, col)
    X_test[col] = impute_na_numerical(X_test, col, mapper_median)

print('===== after missing imputation ======')
print('missing values:')
print(X_test[vars_num_with_na].isnull().sum())
print('missing indicator:')
print(X_test[[var + '_na' for var in vars_num_with_na]].sum())

# ======================================================
# ======================================================
# ============================= Categorical Missing ================================ #
def impute_na_cat(X, col, replacement='Missing'):
    return X[col].fillna(replacement)

vars_cat_with_na = list({

    *[var for var in vars_cat if X_train[var].isnull().any()],
    *[var for var in vars_cat if X_train[var].isnull().any()]
})

print('========missing========')
print('========train========')
for col in vars_cat_with_na:
    print(f'{col}: {X_train[col].isnull().sum()}')
print('========test========')
for col in vars_cat_with_na:
    print(f'{col}: {X_test[col].isnull().sum()}')

for col in vars_cat:
    X_train[col] = impute_na_cat(X_train, col)
    X_test[col] = impute_na_cat(X_test, col)

print('========missing========')
print('========train========')
for col in vars_cat_with_na:
    print(f'{col}: {X_train[col].isnull().sum()}')
print('========test========')
for col in vars_cat_with_na:
    print(f'{col}: {X_test[col].isnull().sum()}')


# ==========================================================================
#  ========== Remove rare labels in categorical variables =================
# =========================================================================

def get_frequent_labels(X, col, th=0.05):
    labels = X[col].value_counts(normalize=True)
    frequent_labels = labels[labels > th].index.to_list()
    return frequent_labels

def replace_rare_labels(X, col, frequent_labels, rare_label='Rare'):
    return np.where(X[col].isin(frequent_labels[col]), X[col], rare_label)


joblib.dump(
    {col: get_frequent_labels(X_train, col) for col in vars_cat},
    'frequent_labels.pkl'
)
frequent_labels = joblib.load('frequent_labels.pkl')
import json
print('train')
for var in vars_cat:
    print(f'=========={var}=========')
    print(f'==========before=========')
    print(json.dumps(X_train[var].value_counts(normalize=True).round(3).to_dict(), indent=-1))
    print(f'=======================')

    X_train[var] = replace_rare_labels(X_train, var, frequent_labels, rare_label='Rare')

    print(f'==========after=========')
    print(json.dumps(X_train[var].value_counts(normalize=True).round(3).to_dict(), indent=-1))
    print(f'=======================')


print('test')
for var in vars_cat:
    print(f'=========={var}=========')
    print(f'==========before=========')
    print(json.dumps(X_test[var].value_counts(normalize=True).round(3).to_dict(), indent=-1))
    print(f'=======================')

    X_test[var] = replace_rare_labels(X_test, var, frequent_labels, rare_label='Rare')

    print(f'==========after=========')
    print(json.dumps(X_test[var].value_counts(normalize=True).round(3).to_dict(), indent=-1))
    print(f'=======================')

# ==========================================================================
#  ========== Perform One Hot encoding =================
# =========================================================================

from feature_engine.categorical_encoders import OneHotCategoricalEncoder
# X=X_train.select_dtypes(include=[object])

for var in vars_cat:
    print(var)
    print(json.dumps(X_train[var].value_counts(normalize=True).round(3).to_dict(), indent=4))

encoder = OneHotCategoricalEncoder(
    top_categories=None,
    variables=vars_cat,
    drop_last=True,
)

encoder=encoder.fit(X_train)
encoder.encoder_dict_
# train
X_train = encoder.transform(X_train)
# test
X_test = encoder.transform(X_test)



# ==========================================================================
#  ========== Scale the data =================
# TODO: Scale the data
# =========================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=True, with_std=True)

scaler = scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# train
print('train')
print(X_train.mean().mean())
print(X_train.std().mean())
print('============')

# test
print('test')
print(X_test.mean().mean())
print(X_test.std().mean())
print('============')



# ==========================================================================
#  ========== Train the model =================
# TODO: Train the logistic regression
# =========================================================================

model = LogisticRegression(C=0.0005, random_state=0)
model = model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')

# ==========================================================================
#  ========== Make prediction and evaluate model performance =================
# TODO: Determine roc-auc, accuracy

# train
train_acc = accuracy_score(y_train, model.predict(X_train))
train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
print(f'Train accuracy: {train_acc}')
print(f'Train roc_auc: {train_roc_auc}')

# test
test_acc = accuracy_score(y_test, model.predict(X_test))
test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f'Train accuracy: {test_acc}')
print(f'Train roc_auc: {test_roc_auc}')



