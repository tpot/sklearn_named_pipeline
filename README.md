# sklearn_named_pipeline

I find it frustrating that scikit-learn preprocessing steps convert pandas dataframes into numpy arrays. Especially when used in pipelines or columnTransformers, column names can be lost and it can be hard to track down what variables correspond to which columns after preprocessing and pipeline transformations. I have written a few transformation classes based on (inheriting) scikit-learn transformations. I hope to add to these as I use more of the transformations.

Scikit-learn transformations generally have a `feature_names_in` attribute, but no `get_feature_names_out` method. The derived classes in classes.py include such a method, which can be chained by `ColumnTransformerNamed`.
