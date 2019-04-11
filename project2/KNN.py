import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics


start_time = time.time()


total_train_features = np.load("train_features.npy")
total_train_labels = np.load("train_labels.npy")
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy")


# k-fold validation
kfold= KFold(n_splits=3,random_state =None)


# metricslist = ["euclidean", "manhattan", "chebyshev"]
metricslist = ["euclidean", "manhattan"]
#range of k
kmax = 15
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

            neigh = KNeighborsClassifier(n_neighbors=k, metric=knn_metrics)
            neigh.fit(train_features, train_labels)
            predict = neigh.predict(valid_features)
            accuracy = metrics.accuracy_score(valid_labels, predict)
            ac_list.append(accuracy)
            print("metric:{} accuracy:{}".format(knn_metrics, accuracy))

        final_train_accuracy = np.mean(ac_list)
        train_ac_metrics_list.append([knn_metrics, k, final_train_accuracy])


train_ac_metrics_list = np.asarray(train_ac_metrics_list)
max_index = np.argmax(train_ac_metrics_list[:, 2])
print(train_ac_metrics_list[max_index, :])


train_ac_metrics_list = pd.DataFrame(train_ac_metrics_list)
train_ac_metrics_list.to_csv("train_metrics_ac_list.csv",index=False, sep=',')
finish_time = time.time()
print("time consumption",finish_time-start_time)