from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
class SimpleImputerNamed(SimpleImputer):
    from sklearn.impute import SimpleImputer
    def get_feature_names_out(self):
        return list(self.feature_names_in_)
class OrdinalEncoderNamed(OrdinalEncoder):
    def get_feature_names_out(self):
        return list(self.feature_names_in_)
class OneHotEncoderNamed(OneHotEncoder):
    def get_feature_names_out(self):
        names_out = []
        for i, name_in in enumerate(self.feature_names_in_):
            names_out += [f'{name_in}_{j}' for j in self.categories_[i]]
        return names_out
class ColumnTransformerNamed(ColumnTransformer):
    def get_feature_names_out(self):
        names = []
        for transformer in self.transformers_:
            if transformer[0] == 'remainder':
                if transformer[1] == 'passthrough':
                    names += list(self.feature_names_in_[transformer[2]])
                break
            else:
                names += transformer[1].get_feature_names_out()
        return names
    def fit(self, X, y=None):
        #print('In fit method')
        return super().fit(X,y)
    def transform(self, X):
        #print('In transform method')
        transformed = super().transform(X)
        return pd.DataFrame(transformed, columns= self.get_feature_names_out())
    def fit_transform(self, X, y=None):
        #print('In fit_transform method')
        fit_transformed = super().fit_transform(X,y)
        return pd.DataFrame(fit_transformed, columns=self.get_feature_names_out())
    


class Identity(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return np.array(X)
class IdentityNamed(Identity):
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self
    def get_feature_names_out(self):
        return self.feature_names_in_
    
class FeatureUnionNamed(FeatureUnion, TransformerMixin):
    def __init__(self, transformer_list):
        self.transformers_ = transformer_list
        super().__init__(transformer_list)
    def get_feature_names_out(self):
        names = []
        for transformer in self.transformers_:
            names += transformer[1].get_feature_names_out()
        return names
    def fit(self, X, y=None):
        return super().fit(X,y)
    def transform(self, X):
        transformed = super().transform(X)
        return pd.DataFrame(transformed, columns= self.get_feature_names_out())
    def fit_transform(self, X, y=None):
        print(X.shape)
        fit_transformed = super().fit_transform(X,y)
        return pd.DataFrame(fit_transformed, columns=self.get_feature_names_out())
    
