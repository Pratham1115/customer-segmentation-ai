from sklearn.preprocessing import StandardScaler

def scale_features(data):

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)

    return scaled_data