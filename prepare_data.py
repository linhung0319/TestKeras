#!/usr/bin/env python2

from sklearn import datasets
from sklearn.model_selection import train_test_split

def generate_data(datasets_name, test_size=0.33, random_state=0):
    if (datasets_name.lower() == 'iris'):
        data = datasets.load_iris()
    x_tr, x_te, y_tr, y_te = train_test_split(data.data,
                                              data.target,
                                              test_size=test_size,
                                              random_state=random_state)
    return x_tr, x_te, y_tr, y_te

if __name__ == '__main__':
    main()
