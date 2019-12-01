import numpy as np


# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)  # formula for derivative of output of sigmoid
    return 1 / (1 + np.exp(-x))


def three_layer_nn():
    # inputs
    X = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]])

    # outputs
    y = np.array([[0], [1], [1], [0]])

    # set random seed and initialize weights
    np.random.seed(1)

    # weights of two different layers
    # randomly initialize with mean 0
    syn0 = 2 * np.random.random((3, 4)) - 1
    syn1 = 2 * np.random.random((4, 1)) - 1

    for i in range(60000):
        # feed forward
        l0 = X
        # l1 = nonlin(np.dot(X, syn0))
        l1 = nonlin(np.dot(l0, syn0))
        l2 = nonlin(np.dot(l1, syn1))

        # calculate error
        out_error = y - l2
        # np_error = np.mean(np.abs(out_error))
        # print(np_error)
        if i % 10000 == 0:
            print(f'Error => {np.mean(np.abs(out_error))}')

        l2_delta = out_error * nonlin(l2, True)

        # how much each value contributes to error
        # back propagation
        l1_error = l2_delta.dot(syn1.T)

        # now go for first layer error
        l1_delta = l1_error * nonlin(l1, True)

        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    print("Output After Training")
    print(l2)


three_layer_nn()
