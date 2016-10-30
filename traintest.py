from sklearn.cross_validation import train_test_split

X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data, train_target, test_size=0.4, random_state=0)