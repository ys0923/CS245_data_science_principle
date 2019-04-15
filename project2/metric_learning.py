import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from metric_learn import LMNN
from metric_learn.mmc import MMC_Supervised
from metric_learn import LSML_Supervised
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

start_time = time.time()
total_train_features = np.load("train_features.npy")
total_train_labels = np.load("train_labels.npy")
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy")


# k-fold validation
kfold= KFold(n_splits=5,random_state =None)

knn_k_list = [5, 10, 15]
train_ac_metrics_list = []
for knn_k in knn_k_list:
    ac_list = []
    # ac_list_orig = []
    fold_cnt = 0
    for train_index, valid_index in kfold.split(total_train_features, total_train_labels):
        train_features = total_train_features[train_index]
        train_labels = total_train_labels[train_index]
        valid_features = total_train_features[valid_index]
        valid_labels = total_train_labels[valid_index]

        fold_cnt += 1
        print("k:", knn_k)
        print("fold:", fold_cnt)
        print("train features shape:", train_features.shape)
        print("train labels shape:", train_labels.shape)
        print("valid features shape:", valid_features.shape)
        print("valid labels shape:", valid_labels.shape)

        # lmnn = LMNN(k=5)
        # transformed_features = lmnn.fit_transform(train_features, train_labels)
        mmc = MMC_Supervised(num_constraints=200)
        transformed_features = mmc.fit_transform(train_features, train_labels)
        neigh = KNeighborsClassifier(n_neighbors=knn_k)
        neigh.fit(transformed_features, train_labels)
        # neigh_orig = KNeighborsClassifier(n_neighbors=knn_k)
        # neigh_orig.fit(train_features, train_labels)
        # predict = neigh.predict(lmnn.transform(valid_features))
        predict = neigh.predict(mmc.transform(valid_features))
        # predict_orig = neigh_orig.predict(valid_features)
        accuracy = metrics.accuracy_score(valid_labels, predict)
        # accuracy_orig = metrics.accuracy_score(valid_labels, predict_orig)
        print("accuracy after metric learning:{}".format(accuracy))
        # print("accuracy before metric learning:{}".format(accuracy_orig))
        ac_list.append(accuracy)
        # ac_list_orig.append(accuracy_orig)

    final_train_accuracy = np.mean(ac_list)
    print(final_train_accuracy)
    # final_train_accuracy_orig = np.mean(ac_list_orig)
    # print(final_train_accuracy_orig)
    # train_ac_metrics_list.append([knn_k, final_train_accuracy, final_train_accuracy_orig])
    train_ac_metrics_list.append([knn_k, final_train_accuracy])

train_ac_metrics_list = np.asarray(train_ac_metrics_list)
max_index = np.argmax(train_ac_metrics_list[:, 1])
print(train_ac_metrics_list[max_index, :])

train_ac_metrics_list = pd.DataFrame(train_ac_metrics_list)
train_ac_metrics_list.to_csv("metric_learn_train_metrics_ac_list.csv",index=False, sep=',')
finish_time = time.time()
print("time consumption",finish_time-start_time)

