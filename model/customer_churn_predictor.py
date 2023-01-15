# package import
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector as selector

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# self-defined function
def diff_month(x):
    d1 = date.today()
    d2 = datetime.strptime(x,'%Y-%m-%d')
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def object2timestamp(x):
    # return time into hour during the day    
    return datetime.strptime(x,'%H:%M:%S').timetuple().tm_hour


# data load
df = pd.read_csv('churn.csv')
df.drop(['Unnamed: 0'], axis=1, inplace = True)



# feature engineering
df['days_since_last_login'] = df.days_since_last_login.map(lambda x: np.nan if int(x) < 0 else int(x))
df['avg_time_spent'] = df.avg_time_spent.map(lambda x: np.nan if float(x) < 0 else float(x))
df['avg_frequency_login_days'] = df.avg_frequency_login_days.map(lambda x: np.nan if x == 'Error' or float(x) < 0 else float(x))
df['points_in_wallet'] = df.points_in_wallet.map(lambda x: np.nan if float(x) < 0 else float(x))
df['membership_period_month'] = df.joining_date.map(diff_month)
df['last_visit_time'] = df.last_visit_time.map(object2timestamp)

    
# target 
Target = 'churn_risk_score'
y = df[Target]
X = df.drop([Target], axis=1)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# numerical data and catagorical data
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(X_train)
categorical_columns = categorical_columns_selector(X_train)

numerical_columns = set(numerical_columns) - {'last_visit_time'}
categorical_columns = set(categorical_columns) - {'security_no', 'joining_date', 'referral_id','offer_application_preference'}


## construct pipeline
# preprocesser
numeric_transformer = Pipeline(
    steps=[("imputer_num", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[("imputer_cat", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, list(numerical_columns)),
        ("cat", categorical_transformer, list(categorical_columns)),
    ]
)

# combine preprocesser with model
steps = [('preprocess', preprocessor),
         ('gbr', GradientBoostingClassifier(learning_rate = 0.1, max_depth = 5, 
                                            min_samples_leaf = 90, min_samples_split = 2, n_estimators = 100))]

model = Pipeline(steps)

# fit model
model_gbt = model.fit(X_train, y_train)

# AUC_ROC
# Training data: 0.9870328886017538
# Test data: 0.9764356805427897

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model_gbt, f)
