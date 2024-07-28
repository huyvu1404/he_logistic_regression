def prepare_data(df, target):
    feature_names = df.columns.tolist()
    feature_names.remove(target)
    X = df[feature_names]
    y = df[target]
    return X.values , y
