"""Module for listing down additional custom functions required for production."""

import pandas as pd

from numpy import inf
def binned_median_income(df):
    """Bin the selling price column using quantiles."""
    return pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., inf],
                               labels=[1, 2, 3, 4, 5])

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

    def __init__(self):
        self.ohe_enc = OneHotEncoder()
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
        cat_columns = list(X.select_dtypes('object').columns)
        num_columns = list(X.select_dtypes('number').columns)
        if cat_columns:
            self.ohe_enc.fit(X[cat_columns])
        

        if num_columns:
            self.med_imputer.fit(X[num_columns])

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
        cat_columns = list(X.select_dtypes('object').columns)
       
        num_columns = list(X.select_dtypes('number').columns)
        X_transformed = X.copy()
        if cat_columns:
            cat_data = self.ohe_enc.transform(X[cat_columns])
            cat_df = pd.DataFrame(X[cat_columns], columns=self.ohe_enc.get_feature_names_out(cat_columns))
        else:
            cat_df = pd.DataFrame()

        if num_columns:
            num_data = self.med_imputer.transform(X_transformed[num_columns])
            num_df = pd.DataFrame(num_data, columns=num_columns)
        else:
            num_df = pd.DataFrame()

        X_transformed = pd.concat([cat_df, num_df], axis=1)
        X_transformed['income_cat'] = pd.cut(X_transformed["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])
        X_transformed["rooms_per_household"] = X_transformed["total_rooms"]/X_transformed["households"]
        X_transformed["bedrooms_per_room"] = X_transformed["total_bedrooms"]/X_transformed["total_rooms"]
        X_transformed["population_per_household"]=X_transformed["population"]/X_transformed["households"]
        return X_transformed
    
     def inverse_transform(self, X):
        """
        Reverses the transformations applied during `transform`.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to be inverse transformed.

        Returns
        -------
        X_original : pandas.DataFrame
            The original data before transformations were applied.
        """
        X_original = X.copy()

        # Reverse the categorical transformations
        if self.cat_columns:
            cat_feature_names = self.ohe_enc.get_feature_names_out(self.cat_columns)
            cat_data = X_original[cat_feature_names]
            cat_data = self.ohe_enc.inverse_transform(cat_data)
            cat_data = pd.DataFrame(cat_data, columns=self.cat_columns)
            cat_data = self.simple_imputer.inverse_transform(cat_data)
            X_original[self.cat_columns] = cat_data
            X_original = X_original.drop(columns=cat_feature_names, errors='ignore')

        # Reverse the numerical transformations
        if self.num_columns:
            num_data = X_original[self.num_columns]
            num_data = self.med_imputer.inverse_transform(num_data)
            X_original[self.num_columns] = num_data

        return X_original

## Unit tests
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    # Sample dataset with categorical and numerical columns
    data = {
    'longitude': [-122.23, -122.22, -122.24, -122.25, -122.25],
    'latitude': [37.88, 37.86, 37.85, 37.85, 37.85],
    'housing_median_age': [41.0, 21.0, 52.0, 52.0, 52.0],
    'total_rooms': [880.0, 7099.0, 1467.0, 1274.0, 1627.0],
    'total_bedrooms': [129.0, 1106.0, 190.0, 235.0, 280.0],
    'population': [322.0, 2401.0, 496.0, 558.0, 565.0],
    'households': [126.0, 1138.0, 177.0, 219.0, 259.0],
    'median_income': [8.3252, 8.3014, 7.2574, 5.6431, 3.8462],
    'ocean_proximity': ['NEAR BAY', 'NEAR BAY', 'NEAR BAY', 'NEAR BAY', 'NEAR BAY']
    }

# Create the DataFrame
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def expected_transformed_data():
    # Expected output after transformation
    data = {
    'ocean_proximity_<1H OCEAN': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'ocean_proximity_INLAND': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'ocean_proximity_ISLAND': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'ocean_proximity_NEAR BAY': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'ocean_proximity_NEAR OCEAN': [np.nan, np.nan, np.nan, np.nan, np.nan],
    'longitude': [-122.23, -122.22, -122.24, -122.25, -122.25],
    'latitude': [37.88, 37.86, 37.85, 37.85, 37.85],
    'housing_median_age': [41.0, 21.0, 52.0, 52.0, 52.0],
    'total_rooms': [880.0, 7099.0, 1467.0, 1274.0, 1627.0],
    'total_bedrooms': [129.0, 1106.0, 190.0, 235.0, 280.0],
    'population': [322.0, 2401.0, 496.0, 558.0, 565.0],
    'households': [126.0, 1138.0, 177.0, 219.0, 259.0],
    'median_income': [8.3252, 8.3014, 7.2574, 5.6431, 3.8462],
    'income_cat': [5, 5, 5, 4, 3],
    'rooms_per_household': [6.984127, 6.238137, 8.288136, 5.817352, 6.281853],
    'bedrooms_per_room': [0.146591, 0.155797, 0.129516, 0.184458, 0.172096],
    'population_per_household': [2.555556, 2.109842, 2.802260, 2.547945, 2.181467]
    }

    # Create the DataFrame
    df = pd.DataFrame(data)
    return df

def test_fit(sample_data):
    transformer = CustomTransformer()
    transformer.fit(sample_data)

def test_transform(sample_data, expected_transformed_data):
    transformer = CustomTransformer()
    transformed_data = transformer.fit_transform(sample_data)

    # Compare transformed data with expected data
    pd.testing.assert_frame_equal(transformed_data, expected_transformed_data)

def test_no_columns(sample_data):
    # Case where no columns are provided
    transformer = CustomTransformer(cat_columns=[], num_columns=[])
    transformed_data = transformer.fit_transform(sample_data)

    # The transformed data should be empty as no columns are to be transformed
    assert transformed_data.empty


if __name__ == '__main__':
    # standard code-template imports
    from pprint import pprint
    from ta_lib.core.api import (
        create_context, get_dataframe, get_feature_names_from_column_transformer, get_package_path,
        display_as_tabs, string_cleaning, merge_info, initialize_environment,
        list_datasets, load_dataset, save_dataset
    )
    import ta_lib.eda.api as eda
    import os
    import os.path as op
    import shutil

    # standard third party imports
    # import numpy as np
    # import pandas as pd
    # from sklearn.model_selection import train_test_split
    pd.options.mode.use_inf_as_na = True

    config_path = op.join('conf', 'config.yml')
    context = create_context(config_path)
    pprint(list_datasets(context))

    housing = load_dataset(context, 'raw/housing')
    y = housing['median_house_value']
    X = housing.drop('median_house_value',axis = 1)
    t = CustomTransformer()
    t.fit(X, y)
    transf = t.transform(X[:5])
    pd.set_option('display.max_rows', None)  # Display all rows
    pd.set_option('display.max_columns', None)  # Display all columns
    pd.set_option('display.width', None)  # Adjust width to avoid truncation

    
    print(transf)
    print(X[:5])