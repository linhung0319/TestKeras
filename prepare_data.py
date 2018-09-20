#!/usr/bin/env python2

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

def generate_data(datasets_name, test_size=0.33, random_state=0):
    if (datasets_name.lower() == 'iris'):
        data = datasets.load_iris()
    x_tr, x_te, y_tr, y_te = train_test_split(data.data,
                                              data.target,
                                              test_size=test_size,
                                              random_state=random_state)
    return x_tr, x_te, y_tr, y_te

def generate_batch(x, y, batch_size):
    x_n_samples = x.shape[0]
    y_n_samples = y.shape[0]
    assert(x_n_samples == y_n_samples)

    index = np.arange(x_n_samples)
    rs = np.random.RandomState(0)
    rs.shuffle(index)

    pointer = 0
    while True:
        batch_index = index[pointer : min(pointer + batch_size, x_n_samples)]
        pointer += batch_size
        if pointer >= x_n_samples:
            pointer = 0
            rs.shuffle(index)

        yield x[batch_index], y[batch_index]

def main():
    x_tr, x_te, y_tr, y_te = generate_data('iris')
    print('train_label:')
    print(y_tr)
    print('test_label:')
    print(y_te)
    print('')

    iter = 0
    for batch_x, batch_y in generate_batch(x_tr, y_tr, 3):
        print( 'iter {},\t batch_x:'.format(str(iter)) )
        print( batch_x )
        print( 'iter {},\t batch_y:'.format(str(iter)) )
        print( batch_y )
        print('')
        iter += 1
        if (iter == 3):
            break


if __name__ == '__main__':
    main()
