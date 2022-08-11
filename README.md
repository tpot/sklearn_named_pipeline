# sklearn_named_pipeline

I find it frustrating that scikit-learn preprocessing steps convert pandas dataframes into numpy arrays. Especially when used in pipelines or columnTransformers, column names can be lost and it can be hard to track down what variables correspond to which columns after preprocessing and pipeline transformations. 

See also: https://github.com/scikit-learn/scikit-learn/issues/5523

I have written a few transformation classes based on (inheriting) scikit-learn transformations. I hope to add to these as I use more of the transformations.

Scikit-learn transformations generally have a `feature_names_in` attribute, but no `get_feature_names_out` method. The derived classes in classes.py include such a method, which can be chained by `ColumnTransformerNamed`.

Example:

    import pandas as pd
    import seaborn as sns

    titanic_data = sns.load_dataset('titanic')
    titanic_data

    transformed_data = ColumnTransformerNamed(transformers = [('encode_ordinal_variables', OrdinalEncoderNamed(), ['class']),
                                                              ('encode_nominal_variables', OneHotEncoderNamed(), ['sex', 'deck']),
                                                              ('impute_with_median', SimpleImputerNamed(strategy='median'), ['age'])],
                                              remainder='passthrough').fit_transform(titanic_data)
    transformed_data.iloc[:,:15]

which produces a pandas dataframe with appropriately named columns (albeit not in the original order). This can be useful for further data analysis, and also when interpreting output from models (such as XGBoost `feature_importance` results).
