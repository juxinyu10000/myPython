import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn. model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


import warnings
warnings.filterwarnings("ignore")


def mro_test():
    mro = pd.read_csv('/ddhome/tools/yun/mro.csv')
    mro.head()
    mro.info()
    mro.describe(include='all')

    mro = mro.dropna(subset=["srsrp"])
    mro = mro.drop("lteScPUSCHPRBNum", axis=1)
    mro = mro.reset_index(drop=True)    
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    label = mro.label
    label_encoded = encoder.fit_transform(label)
    mro['label_encoded'] = label_encoded
    mro['max_nrsrp'] = mro[['nrsrp1' ,'nrsrp2','nrsrp3','nrsrp4', 'nrsrp5', 'nrsrp6']].max(axis=1)
    mro['min_nrsrp'] = mro[['nrsrp1' ,'nrsrp2','nrsrp3','nrsrp4', 'nrsrp5', 'nrsrp6']].min(axis=1)
    mro['diff_rsrp'] = (mro['srsrp'] - mro['max_nrsrp']).abs()
    same_mod3 = lambda pci1, pci2: int(pci1 % 3 == pci2 % 3)
    same_mod3_cnt = lambda row: same_mod3(row['spci'],
    row['npci1']) + same_mod3(row['spci'], row['npci2']) + same_mod3(row['spci'], row['npci3']) \
    + same_mod3(row['spci'],row['npci4']) + same_mod3(row['spci'], row['npci5']) + same_mod3(row['spci'], row['npci6'])
    mro['same_mod3_cnt'] = mro.apply(same_mod3_cnt, axis=1)
    same_band_cnt = lambda row: int(row['searfcn'] == row['nearfcn1']) + int(row['searfcn'] == row['nearfcn2']) + \
    int(row['searfcn'] == row['nearfcn3']) + int(row['searfcn'] == row['nearfcn4']) + \
    int(row['searfcn'] == row['nearfcn5']) + int(row['searfcn'] == row['nearfcn6'])
    mro['same_band_cnt'] = mro.apply(same_band_cnt, axis=1)
    mro_partition=mro[['seci', 'label','srsrp','lteScPHR','lteScSinrUL','max_nrsrp','min_nrsrp','diff_rsrp','same_mod3_cnt','same_band_cnt','label_encoded']]
    mro_partition.to_csv("./mro_readyprocess.csv",index=False,sep=',')
    
    scaler = StandardScaler()
    #选取需要进行标准化的数据
    mro_partition2 = mro_partition[['srsrp','lteScPHR','lteScSinrUL','max_nrsrp','min_nrsrp','diff_rsrp','same_mod3_cnt','same_band_cnt']]
    #标准化
    features =scaler.fit_transform(mro_partition2)
    #将标准化后的ndarray数据转换为dataframe
    df = pd.DataFrame(features,columns=['srsrp','lteScPHR','lteScSinrUL','max_nrsrp','min_nrsrp','diff_rsrp','same_mod3_cnt','same_band_cnt'])
    #合并数据集
    mro_partition3 = pd.DataFrame(columns=['seci','label','label_encoded','srsrp','lteScPHR','lteScSinrUL','max_nrsrp','min_nrsrp','diff_rsrp','same_mod3_cnt','same_band_cnt'])
    mro_partition3['seci'] = mro_partition['seci']
    mro_partition3['label'] = mro_partition['label']
    mro_partition3['label_encoded'] = mro_partition['label_encoded']
    mro_partition3['srsrp'] = df['srsrp']
    mro_partition3['lteScPHR'] = df['lteScPHR']
    mro_partition3['lteScSinrUL'] = df['lteScSinrUL']
    mro_partition3['max_nrsrp'] = df['max_nrsrp']
    mro_partition3['min_nrsrp'] = df['min_nrsrp']
    mro_partition3['diff_rsrp'] = df['diff_rsrp']
    mro_partition3['same_mod3_cnt'] = df['same_mod3_cnt']
    mro_partition3['same_band_cnt'] = df['same_band_cnt']
    mro_partition3.to_csv("./mro_scalerdown.csv",index=False,sep=',')
    mro_scaler = pd.read_csv('./mro_scalerdown.csv')
    mro_scaler.head(5)
    mro_label = mro_scaler. label_encoded!= 2
    
    features = mro_scaler.iloc[:,3:]
    features_train, features_test, labels_train, labels_test = train_test_split(features,mro_label,test_size=0.3, random_state=0)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("训练集Tag的正样本比例：", labels_train.mean())
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print("测试集Tag的正样本比例：", labels_test.mean())
    print('Train Finished!')
    
    lr = LogisticRegression()
    lr.fit(features_train, labels_train)
    
    y_pred = lr.predict(features_test)
    #features_train, features_test, labels_train, labels_test
    
    #Compute confusion matrix
    cnf_matrix = confusion_matrix(labels_test, y_pred)
    np.set_printoptions(precision=2)
    
    print('------------混淆矩阵-------------')
    print(cnf_matrix)
    P = cnf_matrix[1, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1])
    R = cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1])
    F1 = 2 * P * R / (P + R)
    print("P值 in the testing dataset: ", P)
    print("Recall metric in the testing dataset: ", R)
    print("F1 metric in the testing dataset: ", F1)
    
    y_pred_proba = lr.predict_proba(features_test)
    fpr, tpr, thresholds = roc_curve(labels_test, y_pred_proba[:, 1])
    
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    print(roc_auc)
    features = mro_scaler.iloc[:,3:]
    labels = mro_scaler['label_encoded']
    
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3, random_state=0)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print('Train Finished!')
    lr = LogisticRegression(multi_class='multinomial',solver='newton-cg')
    lr.fit(features_train, labels_train)
    #分类报告：precision/recall/fi-score/均值/分类个数
        
    y_true = labels_test
    y_pred = lr.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    
    #将标签二值化
    y = label_binarize(labels_test, classes=[0,1,2,3,4])
    #设置种类
    n_classes = y.shape[1]
    #Learn to predict each class against the other
    classifier = OneVsRestClassifier(lr)
    y_score = classifier.fit(features_train, labels_train).decision_function(features_test)
    
    #计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    #Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    #Compute macro-average ROC curve and ROC area
    #First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    #Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
    model_to_set = OneVsRestClassifier(LogisticRegression(multi_class='multinomial',solver='newton-cg'))
    parameters = {
    "estimator__C": [0.1,1,10,40,60],
    }
    model_tunning = GridSearchCV(model_to_set, param_grid = parameters)
    model_tunning.fit(features_train, labels_train.values.ravel())
    print(model_tunning.best_score_)
    print(model_tunning.best_params_)
    model_to_set = OneVsRestClassifier(LogisticRegression(multi_class='multinomial',solver='newton-cg'))
    parameters = {
    "estimator__C": [30,32,35,40,45],
    }
    model_tunning = GridSearchCV(model_to_set, param_grid = parameters)
    model_tunning.fit(features_train, labels_train.values.ravel())
    print(model_tunning.best_score_)
    print(model_tunning.best_params_)
    lr = LogisticRegression(multi_class='multinomial',solver='newton-cg',C=32)
    lr.fit(features_train, labels_train.values.ravel())
        
    y_true = labels_test
    y_pred = lr.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    mro_data = pd.read_csv('./mro_readyprocess.csv')
    mro_data.head(5)
    features = mro_data [['srsrp','lteScPHR','lteScSinrUL','max_nrsrp','min_nrsrp','diff_rsrp','same_mod3_cnt','same_band_cnt']]
    labels = mro_data[['label_encoded']]
    
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3, random_state=3)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print('Train Finished!')
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(features_train, labels_train)
    y_true = labels_test
    y_pred = tree_clf.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    
    model_to_set = OneVsRestClassifier(DecisionTreeClassifier())
    parameters = {
    "estimator__max_depth": [10,11,12],
    "estimator__min_samples_split": [10,15,20,25],
    "estimator__min_samples_leaf": [10,15,20,25],    
    }
    model_tunning = GridSearchCV(model_to_set, param_grid = parameters)
    model_tunning.fit(features_train, labels_train.values.ravel())
    print(model_tunning.best_score_)
    print(model_tunning.best_params_)
    tree_clf = DecisionTreeClassifier(max_depth=11,min_samples_leaf=15,min_samples_split=20)
    tree_clf.fit(features_train, labels_train)
    
    y_true = labels_test
    y_pred = tree_clf.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    mro_data = pd.read_csv('./mro_readyprocess.csv')
    mro_data.head(5)
    
    features = mro_data [['srsrp','lteScPHR','lteScSinrUL','max_nrsrp','min_nrsrp','diff_rsrp','same_mod3_cnt','same_band_cnt']]
    labels = mro_data[['label_encoded']]
    
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3, random_state=3)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print('Train Finished!')
    
    lr_clf = LogisticRegression(multi_class='multinomial',solver='newton-cg',C=10)
    dt_clf = DecisionTreeClassifier(max_depth=10)
    svm_clf = SVC(random_state=42)
    voting_clf = VotingClassifier(estimators=[('lr', lr_clf), ('dt', dt_clf), ('svc',svm_clf)],voting='hard')
    voting_clf.fit(features_train, labels_train.values.ravel())
    
    y_true = labels_test
    y_pred = voting_clf.predict(features_test) 
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    
    lr_clf = LogisticRegression(multi_class='multinomial',solver='newton-cg',C=10)
    dt_clf = DecisionTreeClassifier(max_depth=10)
    svm_clf = SVC(random_state=42,probability=True)
    soft_voting_clf = VotingClassifier(estimators=[('lr', lr_clf), ('dt', dt_clf), ('svc',svm_clf)],voting='soft')
    soft_voting_clf.fit(features_train, labels_train.values.ravel())
    
    
    y_true = labels_test
    y_pred = soft_voting_clf.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    
    bag_clf = BaggingClassifier(DecisionTreeClassifier(),n_estimators=500, bootstrap=True, n_jobs=1,oob_score=True)
    bag_clf.fit(features_train, labels_train.values.ravel())
    
    y_true = labels_test
    y_pred = bag_clf.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    model_to_set = OneVsRestClassifier(BaggingClassifier(DecisionTreeClassifier(max_depth=10,min_samples_leaf=15,min_samples_split=25),n_jobs=1,oob_score=True))
    parameters = {
    "estimator__n_estimators": [100,125,150,175,200],
    "estimator__bootstrap": ['True','false'],  
    }
    model_tunning = GridSearchCV(model_to_set, param_grid = parameters)
    model_tunning.fit(features_train, labels_train.values.ravel())
    print(model_tunning.best_score_)
    print(model_tunning.best_params_)
    
    bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=10,min_samples_leaf=15,min_samples_split=25),n_estimators=125, bootstrap=True, n_jobs=1,oob_score=True)
    bag_clf.fit(features_train, labels_train.values.ravel())
    
    #分类报告：precision/recall/fi-score/均值/分类个数
        
    y_true = labels_test
    y_pred = bag_clf.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    
    rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16, n_jobs=1)
    rnd_clf.fit(features_train, labels_train.values.ravel())
    y_true = labels_test
    y_pred = rnd_clf.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    model_to_set = OneVsRestClassifier(RandomForestClassifier(n_jobs=-1))
    parameters = {
    "estimator__n_estimators": [100,125,150,175],
    "estimator__max_depth": [9,10,11], 
    }
    model_tunning = GridSearchCV(model_to_set, param_grid = parameters)
    model_tunning.fit(features_train, labels_train.values.ravel())
    print(model_tunning.best_score_)
    print(model_tunning.best_params_)
    
    rnd_clf = RandomForestClassifier(n_estimators=125,max_depth=10, n_jobs=1)
    rnd_clf.fit(features_train, labels_train.values.ravel())
    
    y_true = labels_test
    y_pred = rnd_clf.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    
    
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200,algorithm="SAMME.R", learning_rate=0.5)
    ada_clf.fit(features_train, labels_train.values.ravel())
    
    y_true = labels_test
    y_pred = ada_clf.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    model_to_set = OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=9),algorithm="SAMME.R"))
    parameters = {
    "estimator__n_estimators": [100,125,150,175],
    "estimator__learning_rate": [0.5,1,5], 
    }
    model_tunning = GridSearchCV(model_to_set, param_grid = parameters)
    model_tunning.fit(features_train, labels_train.values.ravel())
    print(model_tunning.best_score_)
    print(model_tunning.best_params_)
    
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9), n_estimators=125,algorithm="SAMME.R", learning_rate=1)
    ada_clf.fit(features_train, labels_train.values.ravel())
    
    y_true = labels_test
    y_pred = ada_clf.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    
    gbdt = GradientBoostingClassifier(max_depth=9,n_estimators=3, learning_rate=1.0)
    gbdt.fit(features_train, labels_train.values.ravel())
    
    y_true = labels_test
    y_pred = gbdt.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    model_to_set = OneVsRestClassifier(GradientBoostingClassifier())
    parameters = {
    "estimator__n_estimators": [150,175,200,250],
    "estimator__learning_rate": [0.05,0.5,1,5], 
    "estimator__max_depth": [3,4,5,6], 
    }
    model_tunning = GridSearchCV(model_to_set, param_grid = parameters)
    model_tunning.fit(features_train, labels_train.values.ravel())
    print(model_tunning.best_score_)
    print(model_tunning.best_params_)
    
    gbdt = GradientBoostingClassifier(max_depth=4,n_estimators=200,learning_rate=0.05)
    gbdt.fit(features_train, labels_train.values.ravel())
    
    y_true = labels_test
    y_pred = gbdt.predict(features_test)
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4']
    print(classification_report(y_true, y_pred, target_names=target_names))


























