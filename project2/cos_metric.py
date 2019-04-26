import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.spatial.distance import cosine
import heapq


def cos_knn(k, test_data, test_target, stored_data, stored_target):
    """k: number of neighbors to use for voting
    test_data: a set of unobserved images to classify
    test_target: the labels for the test_data (for calculating accuracy)
    stored_data: the images already observed and available to the model
    stored_target: labels for stored_data
    """

    # find cosine similarity for every point in test_data between every other point in stored_data
    cosim = cosine_similarity(test_data, stored_data)

    # get top k indices of images in stored_data that are most similar to any given test_data point
    top = [(heapq.nlargest((k), range(len(i)), i.take)) for i in cosim]
    # convert indices to numbers using stored target values
    top = [[stored_target[j] for j in i[:k]] for i in top]

    # vote, and return prediction for every image in test_data
    pred = [max(set(i), key=i.count) for i in top]
    pred = np.array(pred)

    return pred



start_time = time.time()


total_train_features = np.load("train_features.npy")
total_train_labels = np.load("train_labels.npy")
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy")
# total_train_features = preprocessing.normalize(total_train_features, norm='l2')
# total_train_labels = preprocessing.normalize(total_train_labels, norm='l2')
# test_features = preprocessing.normalize(test_features)
# test_labels = preprocessing.normalize(test_labels)

# k-fold validation
kfold= KFold(n_splits=5,random_state =None)


# metricslist = ["euclidean", "manhattan", "chebyshev"]
metricslist = ["euclidean"]
#range of k
kmax = 20
kmin = 5
numk = (kmax - kmin)/5
train_ac_metrics_list = []
for knn_metrics in metricslist:
    for k in range(kmin, kmax, 5):
        ac_list = []
        fold_cnt = 0
        for train_index,valid_index in kfold.split(total_train_features,total_train_labels):
            train_features = total_train_features[train_index]
            train_labels = total_train_labels[train_index]
            valid_features = total_train_features[valid_index]
            valid_labels = total_train_labels[valid_index]

            fold_cnt += 1
            print("k:", k)
            print("fold:", fold_cnt)
            print("train features shape:",train_features.shape)
            print("train labels shape:",train_labels.shape)
            print("valid features shape:",valid_features.shape)
            print("valid labels shape:", valid_labels.shape)


            predict = cos_knn(k, valid_features, valid_labels, train_features, train_labels)
            accuracy = metrics.accuracy_score(valid_labels, predict)
            ac_list.append(accuracy)
            print("metric:{} accuracy:{}".format(knn_metrics, accuracy))

        final_train_accuracy = np.mean(ac_list)
        train_ac_metrics_list.append([knn_metrics, k, final_train_accuracy])


train_ac_metrics_list = np.asarray(train_ac_metrics_list)
max_index = np.argmax(train_ac_metrics_list[:, 2])
print(train_ac_metrics_list[max_index, :])


train_ac_metrics_list = pd.DataFrame(train_ac_metrics_list)
train_ac_metrics_list.to_csv("cos_train_metrics_ac_list.csv",index=False, sep=',')
finish_time = time.time()
print("time consumption",finish_time-start_time)