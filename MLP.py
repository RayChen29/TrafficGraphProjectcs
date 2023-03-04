import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.neural_network import MLPClassifier
import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# PROGRAM DEFINITIONS
PANDAS_AXIS_COLUMN = 1
PANDAS_AXIS_ROWS = 0
NUMPY_AXIS_COLUMN = 1
NUMPY_AXIS_ROW = 0
DATA_ZERO = 0.01
CONFUSION_MATRIX = np.zeros((2, 2), dtype=int)
NUM_ETA = 0.1
# NUM_ETA = 0.01
# NUM_ETA = 0.001
# NUM_MOMENTUM = 0
# NUM_MOMENTUM = 0.25
NUM_MOMENTUM = 0.5
# NUM_MOMENTUM = 0.9
N = 2  # nodes in hidden layer
NUM_NODES_HIDDEN = N + 1  # nodes in hidden layer + bias node
NUM_NODES_OUTPUT = 1
VEC_ONES_OUTPUT = np.ones(NUM_NODES_OUTPUT, dtype=float)
VEC_ONES_HIDDEN = np.ones(NUM_NODES_HIDDEN, dtype=float)
# MATRIX_OF_PREV_DELTA_WEIGHTS_HID_TO_OUT = np.zeros((NUM_NODES_OUTPUT, NUM_NODES_HIDDEN),
#                                                   dtype=float)  # 10 x N + 1 matrix
# MATRIX_OF_PREV_DELTA_WEIGHTS_IN_TO_HID = np.zeros((N, 9), dtype=float)  # N x 9 matrix
MATRIX_TILE_HID = np.zeros((NUM_NODES_OUTPUT, NUM_NODES_HIDDEN), dtype=float)  # 10 x N + 1 matrix
MATRIX_TILE_IN = np.zeros((N, 9), dtype=float)  # N x 9


def relu(x):
    return np.maximum(0.0, x)


def sigmoid(x):
    """
    Calculates sigmoid activations for hidden/output layers.
    :param x: a vector
    :return: vector with sigmoid function applied to all elements.
    """
    return 1.0 / (1.0 + np.exp(-x))  # assumes x is not arbitrarily large or small.


def get_weights(train_arr):
    w_hid = np.random.uniform(-0.05, 0.05, (N, len(train_arr[0])))
    w_out = np.random.uniform(-0.05, 0.05, (NUM_NODES_OUTPUT, NUM_NODES_HIDDEN))
    return w_hid, w_out


def get_train_data():
    td = np.loadtxt('titanic/pp_train.csv', delimiter=',', dtype=float, skiprows=1)
    # remove columns with pandas library indexing for train and test data
    td = np.delete(td, 0, 1)
    # remove passenger id column in train data
    td = np.delete(td, 0, 1)

    np.savetxt("titanic/raw_train_data.csv", td, delimiter=",")
    # get the survival labels
    labels_train = np.zeros(len(td[:, 0]))
    for idx, item in enumerate(td):
        labels_train[idx] = td[idx][0]

    # replace survival labels with bias
    td[:, 0] = np.ones(len(td[:, 0]))

    # scale the data
    td = td * 1/10
    td[:, 0] = td[:, 0] * 10  # restore bias
    td[:, 3] = td[:, 3] * 1/10  # age in range 0-100
    td[:, 6] = td[:, 6] * 1/100  # fare in range 0-200

    for idx,itm in enumerate(td):  # assert that all date is scaled from 0 to 1
        for i, m in enumerate(itm):
            assert(m <= 1.0)

    # nonzero the data.
    td[:, 1:] = td[:, 1:] + 0.00001
    print(td[0])
    return td, labels_train


def init_hidden():
    arr = np.zeros(NUM_NODES_HIDDEN, dtype=float)
    arr[0] = 1.0
    return arr


def get_test_data():
    """
    :return:
    """
    # for test_data, just add the bias to the first column
    test_data = np.loadtxt('titanic/pp_test.csv', delimiter=',', dtype=float, skiprows=1)
    # test_data = np.delete(test_data, 0, 1)
    np.savetxt("titanic/raw_test_data.csv", test_data, delimiter=",")
    # add bias
    test_data[:, 0] = np.ones(len(test_data[:, 0]))
    # for test data, there are no ground truth labels, therefore one needs to get the passenger ids
    labels_test = np.zeros(len(test_data[:, 0]))
    for idx, item in enumerate(test_data):
        labels_test[idx] = test_data[idx][1]
    # we have the passenger labels, now drop the column from the test data.
    test_data = np.delete(test_data, 1, 1)

    # scale the data
    # scale the data
    test_data = test_data * 1 / 10
    test_data[:, 0] = test_data[:, 0] * 10  # restore bias
    test_data[:, 3] = test_data[:, 3] * 1 / 10  # age in range 0-100
    test_data[:, 6] = test_data[:, 6] * 1 / 100  # fare in range 0-200

    # nonzero the data.
    test_data[:, 1:] = test_data[:, 1:] + 0.00001
    print(test_data[0])
    return test_data, labels_test


def format_csv(file):
    """
    Format Original Data Into A CSV That Can Be Parsed By Numpy
    :param file:
    :return:
    """
    df = pd.read_csv('titanic/' + str(file))
    df = df.drop(columns=['Ticket', 'Name'])  # names and ticket numbers are unnecessary data.
    df["Embarked"].replace({"S": "1.0", "C": "2.0", "Q": "3.0"}, inplace=True)  # map embarked to ints
    df["Sex"].replace({"male": 0.0, "female": 1.0}, inplace=True)  # map sex to ints

    # unused preprocessing if the cabin value is to be included in the final calculations
    mapping = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5", "F": "6", "G": "7", "H": "8", "I": "9",
               "J": "10", "K": "11", "L": "12", "M": "13", "N": "14", "O": "15", "P": "16", "Q": "17", "R": "18",
               "S": "19", "T": "20", "U": "21", "V": "22", "W": "23", "X": "24", "Y": "25", "Z": "26"}
    for k, v in mapping.items():
        df["Cabin"] = df["Cabin"].str.replace(k, v)

    # replace cabin column values with a count of how many cabins that passenger rented
    cabins = df["Cabin"]
    cabins_mapped_to_passenger = []
    for idx, itm in enumerate(cabins):
        if str(itm) == 'nan':  # itm is always non zero, thus check for 'nan'
            cabins_mapped_to_passenger.append(0.0)
        else:
            cabins_mapped_to_passenger.append(str(len(str(itm).split())))
    df["Cabin"] = cabins_mapped_to_passenger  # exchange the values for the cabin column
    df = df.fillna(DATA_ZERO)  # initialize any missing data to a nonzero amount.
    df.to_csv('titanic/pp_' + str(file))


def calculate_deltas(ground_truth, hid_nodes, out_node, w_h_to_o):
    """
    Replace activations of output with delta error values as well as calculate delta error values for hidden layer.
    :param w_h_to_o: weights for the hidden layer to the output layer.
    :param ground_truth: HOT vector for ground truth for a previous propagation.
    :param hid_nodes: activation values for hidden layer
    :param out_node: activation values for output layer
    :return: Pair
    (vector with deltas for hidden layer,
    a vector with deltas for outer layer)
    """
    global VEC_ONES_OUTPUT
    global VEC_ONES_HIDDEN
    # OUTER
    # replace the output activations with delta values.  k = activ_k(1 - activ_k)(ground_truth - activ_k)
    vec_delta_out = out_node * (VEC_ONES_OUTPUT - out_node) * (ground_truth - out_node)

    # INNER
    # scalar calculations and transformations
    bulk_probability_vector = hid_nodes * (VEC_ONES_HIDDEN - hid_nodes)  # calculate (hj (1 - hj))
    temp = np.transpose(w_h_to_o)
    vec_delta_hid = temp.dot(vec_delta_out)

    # multiply the right term and the left term together.
    vec_delta_hid = np.multiply(bulk_probability_vector, vec_delta_hid)  # hj(1-hj) * sigma(w_jk * delta_k)
    return vec_delta_hid, vec_delta_out


def forward_propagate(example, w_h, l_hid, w_o, l_out):
    assert(example[0] == 1.0)
    assert(l_hid[0] == 1.0)
    l_hid[1:] = w_h.dot(example)
    l_hid[1:] = sigmoid(l_hid[1:])
    # l_hid[1:] = relu(l_hid[1:])
    l_out = w_o.dot(l_hid)
    l_out = sigmoid(l_out)
    # l_out = relu(l_out)
    return l_out, l_hid


def get_accuracy_train(train_input, weights_input, hidden, weights_hidden, result, labels):
    correct = 0.0
    incorrect = 0.0
    for idx_gat, x in enumerate(train_input):
        result, unused = forward_propagate(x, weights_input, hidden, weights_hidden, result)
        actual = labels[idx_gat]
        p = result[0]
        if p < 0:
            p = 0
        else:
            p = 1
        if p == actual:
            correct += 1.0
        else:
            incorrect += 1.0
    return correct / (correct + incorrect)


if __name__ == '__main__':
    print("useless")
    # format_csv('train.csv')
    # format_csv('test.csv')
    # train_data, labels_train = get_train_data()  # data
    # test_data, passenger_ids = get_test_data()
    # weights_hidden, weights_output = get_weights(train_data)  # weights
    # prev_delta_w_output = np.zeros((NUM_NODES_OUTPUT, NUM_NODES_HIDDEN), dtype=float)  # 10 x N + 1 matrix
    # prev_delta_w_hidden = np.zeros((N, 9), dtype=float)  # N x 9 matrix
    # layer_hidden = init_hidden()  # nodes in layers
    # layer_out = np.zeros(1, dtype=float)
    # for epoch in range(0, 50):
    #     for idx, item in enumerate(train_data):
    #         # FORWARD PROPAGATE
    #         layer_out, layer_hidden = forward_propagate(train_data[idx], weights_hidden, layer_hidden, weights_output, layer_out)
    #         predicted = layer_out[0]
    #         # DELTAS
    #         delta_h, delta_o = calculate_deltas(labels_train[idx], layer_hidden, layer_out, weights_output)
    #         #
    #         # WEIGHT UPDATES
    #         #
    #         # 1.  HIDDEN TO OUT
    #         #
    #         # calculate Delta_weight_kj = eta * delta_k * hidden_j + momentum * delta_previous_weight_kj
    #         temp_vector = NUM_ETA * delta_o  # ETA * Delta_K for left term.
    #         # momentum * prev delta_weights.
    #         prev_delta_w_output = np.multiply(NUM_MOMENTUM, prev_delta_w_output)
    #         # temp
    #         MATRIX_TILE_HID = np.tile(temp_vector, (NUM_NODES_HIDDEN, 1))
    #         MATRIX_TILE_HID = np.transpose(MATRIX_TILE_HID)
    #         MATRIX_TILE_HID = np.multiply(layer_hidden, MATRIX_TILE_HID)  # eta * delta_k * h_j
    #         MATRIX_TILE_HID = MATRIX_TILE_HID + prev_delta_w_output # eta * delta_k * h_j + a * delta_w_kj
    #         # assignments
    #         weights_output = prev_delta_w_output + MATRIX_TILE_HID  # updated weights
    #         prev_delta_w_output = MATRIX_TILE_HID  # update delta weights
    #         #
    #         # 2.  INPUT TO HIDDEN
    #         #
    #         temp_vector = NUM_ETA * delta_h[1:]  # eta * deltaj.  (skips bias delta)
    #         # momentum * prev delta weights
    #         prev_delta_w_hidden = np.multiply(NUM_MOMENTUM, prev_delta_w_hidden)
    #         MATRIX_TILE_IN = np.tile(temp_vector, (9, 1))
    #         MATRIX_TILE_IN = np.transpose(MATRIX_TILE_IN)
    #         # finish the left term of the formula (eta * deltaj * x_i)
    #         MATRIX_TILE_IN = np.multiply(MATRIX_TILE_IN, train_data[idx])
    #         # with the right term and left term calculated, sum them together to produce the new delta weight matrix.
    #         MATRIX_TILE_IN = MATRIX_TILE_IN + prev_delta_w_hidden  # add momentum matrix
    #         # assignments
    #         weights_hidden = prev_delta_w_hidden + MATRIX_TILE_IN  # update weights
    #         prev_delta_w_hidden = MATRIX_TILE_IN  # update previous delta weights
    #
    #     accur_train = get_accuracy_train(train_data, weights_hidden, layer_hidden, weights_output, layer_out, labels_train)
    #     print(accur_train)


# MLP implementation lifted from code taken from https://scikit-learn.org/stable/modules/neural_networks_supervised.html
train_data, labels = get_train_data()
test_data, ids = get_test_data()
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    max_iter=50,
    alpha=1e-4,
    solver="sgd",
    verbose=10,
    random_state=1,
    learning_rate_init=0.2,
)

with warnings.catch_warnings():
    print("whatever")
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(train_data, labels)

# get the harsh accuracy on the training data for 50 epochs.
harsh_avg = mlp.score(train_data, labels)
print("Training set score: %f" % harsh_avg)
# get recall and precision
pred_train = mlp.predict(train_data)
cm = confusion_matrix(labels, pred_train)
print("CONFUSION MATRIX FOR TRAIN DATA\nS = Survived")
print("\t!S\tS")
print(cm)
# get predictions for test_data
pred = mlp.predict(test_data)
submission = np.zeros((len(test_data), 2), dtype=float)
submission[:, 0] = ids
submission[:, 1] = pred
# package it up to submit to Kaggle
np.savetxt('submission.csv', submission, delimiter=',', fmt='%i')

