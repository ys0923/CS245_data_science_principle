import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics


start_time = time.time()

total_train_features = np.load("train_features.npy")
total_train_labels = np.load("train_labels.npy")
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy")
# k-fold validation
kfold= KFold(n_splits=5,random_state =None)
train_ac_dict = {}
#range of k
kmax = 5
kmin = 3
for k in range(kmin, kmax):
    ac_list = []
    fold_cnt = 0
    for train_index,valid_index in kfold.split(total_train_features,total_train_labels):
        train_features = total_train_features[train_index]
        train_labels = total_train_labels[train_index]
        valid_features = total_train_features[valid_index]
        valid_labels = total_train_labels[valid_index]
        print("test")
        # print(train_features[0])
        print(train_labels[0])
        # print(valid_features[0])
        print(valid_labels[0])


        fold_cnt += 1
        print("k:", k)
        print("fold:", fold_cnt)
        print("train features shape:",train_features.shape)
        print("train labels shape:",train_labels.shape)
        print("valid features shape:",valid_features.shape)
        print("valid labels shape:", valid_labels.shape)

        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_features, train_labels)
        predict = neigh.predict(valid_features)
        accuracy = metrics.accuracy_score(valid_labels, predict)
        ac_list.append(accuracy)
        print("accuracy:{}".format(accuracy))

    final_train_accuracy = np.mean(ac_list)
    train_ac_dict[k] = final_train_accuracy

f = open('train_accuracy_dict.txt','w')
f.write(str(train_ac_dict))
f.close()

max_ac_key = max(train_ac_dict, key=train_ac_dict.get)
print(train_ac_dict)
print(max_ac_key)

finish_time = time.time()
print("time consumption",finish_time-start_time)