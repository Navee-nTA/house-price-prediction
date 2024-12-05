"""Module for listing down additional custom functions required for the notebooks."""

import pandas as pd

def binned_selling_price(df):
    """Bin the selling price column using quantiles."""
    return pd.qcut(df["unit_price"], q=10)


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for applying different transformations to 
    categorical and numerical columns in a DataFrame.

    Parameters
    ----------
    cat_columns : list of str, optional
        The list of categorical column names to be transformed. 
        Default is None.
    
    num_columns : list of str, optional
        The list of numerical column names to be transformed. 
        Default is None.

    Attributes
    ----------
    ohe_enc : OneHotEncoder
        An instance of `OneHotEncoder` used for encoding categorical columns.
    
    simple_imputer : SimpleImputer
        An instance of `SimpleImputer` used for imputing missing values 
        in categorical columns.
    
    med_imputer : SimpleImputer
        An instance of `SimpleImputer` used for imputing missing values 
        in numerical columns.

    Methods
    -------
    fit(X, y=None)
        Fits the transformer on the provided data.

    transform(X)
        Transforms the data according to the fitted transformers.

    fit_transform(X, y=None)
        Fits the transformer and transforms the data in a single step.
    """
    def __init__(self, cat_columns=None, num_columns=None):
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.ohe_enc = OneHotEncoder(sparse_output=False)
        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        self.med_imputer = SimpleImputer(strategy='median')

    def fit(self, X, y=None):
        """
        Fits the transformer on the provided data.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data containing the columns to be transformed.
        
        y : None
            Ignored, present here for API consistency by convention.
        
        Returns
        -------
        self : CustomTransformer
            Fitted transformer.
        """
        if self.cat_columns:
            self.ohe_enc.fit(X[self.cat_columns])
            self.simple_imputer.fit(X[self.cat_columns])

        if self.num_columns:
            self.med_imputer.fit(X[self.num_columns])

        return self

    def transform(self, X):
        """
        Transforms the data according to the fitted transformers.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to be transformed.

        Returns
        -------
        X_transformed : pandas.DataFrame
            The transformed data with encoded categorical columns and 
            imputed numerical columns.
        """
        X_transformed = X.copy()

        if self.cat_columns:
            cat_data = self.simple_imputer.transform(X_transformed[self.cat_columns])
            cat_data = self.ohe_enc.transform(cat_data)
            cat_df = pd.DataFrame(cat_data, columns=self.ohe_enc.get_feature_names_out(self.cat_columns))
        else:
            cat_df = pd.DataFrame()

        if self.num_columns:
            num_data = self.med_imputer.transform(X_transformed[self.num_columns])
            num_df = pd.DataFrame(num_data, columns=self.num_columns)
        else:
            num_df = pd.DataFrame()

        X_transformed = pd.concat([cat_df, num_df], axis=1)

        return X_transformed

##Unit Tests

import pytest
import pandas as pd
import numpy as np
from custom_transformer import CustomTransformer

@pytest.fixture
def sample_data():
    # Sample dataset with categorical and numerical columns
    data = pd.DataFrame({
        'column1': ['A', 'B', np.nan, 'A'],
        'column2': ['X', np.nan, 'Y', 'X'],
        'column3': [1, 2, np.nan, 4],
        'column4': [np.nan, 5, 6, 7]
    })
    return data

@pytest.fixture
def expected_transformed_data():
    # Expected output after transformation
    return pd.DataFrame({
        'column1_A': [1.0, 0.0, 0.0, 1.0],
        'column1_B': [0.0, 1.0, 0.0, 0.0],
        'column1_nan': [0.0, 0.0, 1.0, 0.0],
        'column2_X': [1.0, 0.0, 0.0, 1.0],
        'column2_Y': [0.0, 0.0, 1.0, 0.0],
        'column2_nan': [0.0, 1.0, 0.0, 0.0],
        'column3': [1.0, 2.0, 2.0, 4.0],
        'column4': [6.0, 5.0, 6.0, 7.0]
    })

def test_fit(sample_data):
    cat_columns = ['column1', 'column2']
    num_columns = ['column3', 'column4']
    
    transformer = CustomTransformer(cat_columns=cat_columns, num_columns=num_columns)
    transformer.fit(sample_data)

    # Check that fit stores the correct column names
    assert transformer.cat_columns == cat_columns
    assert transformer.num_columns == num_columns

def test_transform(sample_data, expected_transformed_data):
    cat_columns = ['column1', 'column2']
    num_columns = ['column3', 'column4']
    
    transformer = CustomTransformer(cat_columns=cat_columns, num_columns=num_columns)
    transformed_data = transformer.fit_transform(sample_data)

    # Compare transformed data with expected data
    pd.testing.assert_frame_equal(transformed_data, expected_transformed_data)

def test_no_columns(sample_data):
    # Case where no columns are provided
    transformer = CustomTransformer(cat_columns=[], num_columns=[])
    transformed_data = transformer.fit_transform(sample_data)

    # The transformed data should be empty as no columns are to be transformed
    assert transformed_data.empty
