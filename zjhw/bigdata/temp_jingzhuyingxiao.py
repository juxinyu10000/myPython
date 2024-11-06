# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn. model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline  
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")


def show_score_comparation(_model, X_train, X_test, y_train, y_test, 
                           bar_width=0.2, opacity=0.8, color_train='b', color_test='g', fig_size_w=20, fig_size_h=8, mark_font_size=10):
    
    #通过模型，对训练集和测试集的Feature分别进行预测
    y_predict = _model.predict(X_test)
    y_predict_train = _model.predict(X_train)
    
    #分别为训练集和测试集组织模型评估分数序列，供画图所用
    score_train = (accuracy_score(y_train, y_predict_train), 
                   precision_score(y_train, y_predict_train),
                   recall_score(y_train, y_predict_train),
                   f1_score(y_train, y_predict_train))
    score_test = (accuracy_score(y_test, y_predict), 
                  precision_score(y_test, y_predict),
                  recall_score(y_test, y_predict),
                  f1_score(y_test, y_predict))

    plt.subplot(111)

    index = np.arange(len(score_train))

    rects1 = plt.bar(index, score_train, bar_width,
                     alpha=opacity,
                     color=color_train,
                     label='Train')
    mark_scores(score_train,mark_font_size=mark_font_size)

    rects2 = plt.bar(index + bar_width, score_test, bar_width,
                     alpha=opacity,
                     color=color_test,
                     label='Test')
    mark_scores(score_test, x_offset=bar_width,mark_font_size = mark_font_size)
    
    plt.xlabel('Score Type')
    plt.ylabel('Scores')
    plt.title('Scores Comparation')
    plt.xticks(index + bar_width, ('Accuracy', 'Precision', 'Recall', 'F1'))
    plt.yticks(list(np.arange(0.0,1.0,0.1)))
    plt.legend()
    plt.tight_layout()

    plt.gcf().set_size_inches(fig_size_w, fig_size_h)
    plt.show()

def mark_scores(scores, x_offset=0, mark_font_size=10):
    for each_score_index in range(len(scores)):
        plt.text(each_score_index + x_offset, 
             scores[each_score_index] + 0.05, '%.3f' % scores[each_score_index], 
             ha='center', va= 'bottom',fontsize=mark_font_size)


def plot_feature_importances(tree_model, a_feature_names):
   
    feature_importance_array_type = [('name',object), ('importance', float)]
    feature_importance_array_value = [(name, round(importance * 100, 2)) for name, importance in zip(a_feature_names,tree_model.feature_importances_)]
    feature_importance_array = np.array(feature_importance_array_value, dtype=feature_importance_array_type)
    feature_importance_array = np.sort(feature_importance_array, order='importance')
    
    c_features = len(feature_importance_array)
    plt.barh(range(c_features), [each_importance for name, each_importance in feature_importance_array])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Name")
    plt.yticks(np.arange(c_features), [name for name, each_importance in feature_importance_array],fontsize=10)
    plt.title("Feature Importance")
    for each_feature_index in range(c_features):
        plt.text(feature_importance_array[each_feature_index][1] + 1,each_feature_index, '%.2f%%' % feature_importance_array[each_feature_index][1], ha='center', va= 'center',fontsize=10)
    
    plt.show()


def test_jingzhunyingxiao():

    df = pd.read_csv('/ddhome/tools/yun/markting_datav3.csv')
    df.head(10)
    features=df.iloc[:,0:-1]
    features=features.drop(['ASSET_ROW_ID','is_kdts','is_itv_up','is_mobile_up'],axis=1)
    features_names = features.columns
    labels=df['is_rh_next'] #目标变量
    
    print("Feature的矩阵规模：", features.shape)
    print("Tag的矩阵规模：", labels.shape) 
    #将数据归一到 [ 0，1 ]
    from sklearn import preprocessing
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    print(features)
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3, random_state=0)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("训练集Tag的正样本比例：", labels_train.mean())
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print("测试集Tag的正样本比例：", labels_test.mean())
    print('Train Finished!')
    
    lr = LogisticRegression(class_weight='balanced',solver = 'liblinear')
    lr.fit(features_train, labels_train)
    y_pred = lr.predict(features_test)
    # features_train, features_test, labels_train, labels_test
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(labels_test, y_pred)
    np.set_printoptions(precision=2)
    
    print('------------混淆矩阵-------------')
    print(cnf_matrix)
    print("P值 in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1]))
    print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
    
    plt.figure(figsize=(10,5))
    show_score_comparation(lr,features_train, features_test, labels_train, labels_test)
    plt.show()
    
    gsearch2 = GridSearchCV(estimator = LogisticRegression(class_weight='balanced'),param_grid = {'solver':['liblinear','newton-cg','sag']},n_jobs=1,scoring='roc_auc', cv=3)
    gsearch2.fit(features_train, labels_train)
    print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)
    gsearch2 = GridSearchCV(estimator = LogisticRegression(class_weight='balanced',solver = 'newton-cg'),param_grid = {'C':[0.01,0.1,1,5]},n_jobs=1,scoring='roc_auc', cv=3)
    gsearch2.fit(features_train, labels_train.ravel())
    print(gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_)
    lr = LogisticRegression(C=5, class_weight ='balanced',solver = 'newton-cg')
    lr.fit(features_train, labels_train)
    plt.figure(figsize=(10, 5))
    show_score_comparation(lr, features_train, features_test,labels_train, labels_test)
    plt.show()
    
    features=df.iloc[:,0:-1]
    features=features.drop(['ASSET_ROW_ID','is_kdts','is_itv_up','is_mobile_up'],axis=1)
    labels=df['is_rh_next'] #目标变量
    features = features.drop(['FIBER_ACCESS_CATEGORY', 'BABY_FLG'], axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3, random_state=0)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("训练集Tag的正样本比例：", labels_train.mean())
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print("测试集Tag的正样本比例：", labels_test.mean())
    print('Train Finished!')
    
    lr = LogisticRegression(C=5,class_weight ='balanced',solver = 'newton-cg')
    lr.fit(features_train, labels_train)
    plt.figure(figsize=(10, 5))
    show_score_comparation(lr, features_train, features_test,labels_train, labels_test)
    plt.show()
    
    features=df.iloc[:,0:-1]
    features=features.drop(['ASSET_ROW_ID','is_kdts','is_itv_up','is_mobile_up'],axis=1)
    labels=df['is_rh_next'] #目标变量
    min_max_scaler = preprocessing.MinMaxScaler()
    features = min_max_scaler.fit_transform(features)
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3, random_state=0)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("训练集Tag的正样本比例：", labels_train.mean())
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print("测试集Tag的正样本比例：", labels_test.mean())
    print('Train Finished!')
    lr = LogisticRegression(C=1, class_weight ='balanced',solver = 'newton-cg')
    lr.fit(features_train, labels_train)
    plt.figure(figsize=(10, 5))
    show_score_comparation(lr, features_train, features_test,labels_train, labels_test)
    plt.show()
    
    out = pd.concat([pd.DataFrame(np.array(df.ASSET_ROW_ID)),pd.DataFrame(lr.predict_proba(features))],axis=1)
    out.to_csv("./Probability_total_lr.csv",index=False,sep=',')
    
    path=u'/ddhome/tools/yun'
    df=pd.DataFrame(pd.read_csv(os.path.join(path,'markting_datav3.csv'),encoding='utf-8'))
    df.head(10)
    
    features=df.iloc[:,0:-1]
    features=features.drop(['ASSET_ROW_ID','is_kdts','is_itv_up','is_mobile_up'],axis=1)
    labels=df['is_rh_next'] #目标变量
    
    print("Feature的矩阵规模：", features.shape)
    print("Tag的矩阵规模：", labels.shape)
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3, random_state=0)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("训练集Tag的正样本比例：", labels_train.mean())
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print("测试集Tag的正样本比例：", labels_test.mean())
    print('Train Finished!')
    
    
    clf = DecisionTreeClassifier().fit(features_train, labels_train)
    print('Train Finished!')
    plt.figure(figsize=(10, 5))
    show_score_comparation(clf, features_train, features_test,labels_train, labels_test)
    plt.show()  
    
    pipeline = Pipeline([
        ('clf', DecisionTreeClassifier(class_weight='balanced'))
    ])
    
    parameters = {
        'clf__max_depth': (4,5,6,7,8,9),
        'clf__min_samples_split': (40,50,60),
        'clf__min_samples_leaf': (5,10,15)
    }
    
    model_selection = GridSearchCV(
        pipeline, parameters, n_jobs=-1,scoring='roc_auc')
    model_selection.fit(features_train, labels_train.ravel())
    print('最佳效果：%0.3f' % model_selection.best_score_)
    print('最优参数')
    best_parameters = model_selection.best_best_parameters = model_selection.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name])) 
    
    clf = DecisionTreeClassifier(max_depth=7,class_weight='balanced',min_samples_leaf=10,min_samples_split=60).fit(features_train, labels_train)
    print('Train Finished!')
    plt.figure(figsize=(10, 5))
    show_score_comparation(clf, features_train, features_test,labels_train, labels_test) 
        
    
    feature_name = features_train.columns[0:]
    plt.figure(figsize=(50,70))
    plot_feature_importances(clf, feature_name)    
        
    feature_importance_array_type = [('name',object), ('importance', float)]
    feature_importance_array_value = [(name, round(importance * 100, 2)) for name, importance in zip(feature_name,clf.feature_importances_)]
    feature_importance_array = np.array(feature_importance_array_value, dtype=feature_importance_array_type)
    feature_importance_array = np.sort(feature_importance_array, order='importance')
    print(feature_importance_array)   
        
    features = features[['PROM_TYPES_包3年','STMT_AMT','serv_in_time','SERV_START_DT','DOWN_VOL','DOWN_SPEED','PROM_AMT',
                      'AVG_STMT_AMT','LAST_MONTH_DOWN_VOL','IS_LS']]
    
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.3, random_state=0)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("训练集Tag的正样本比例：", labels_train.mean())
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print("测试集Tag的正样本比例：", labels_test.mean())
    print('Train Finished!')
    print("Feature的矩阵规模：", features.shape)
    print("Tag的矩阵规模：", labels.shape)
    
    
    clf = DecisionTreeClassifier(max_depth=7,class_weight='balanced',min_samples_leaf=10,min_samples_split=60).fit(features_train, labels_train)
    print('Train Finished!')
    plt.figure(figsize=(10, 5))
    show_score_comparation(clf, features_train, features_test,labels_train, labels_test)
    plt.show()
    
    
    features=df.iloc[:,0:-1]
    features=features.drop(['is_kdts','is_itv_up','is_mobile_up'],axis=1)
    labels=df['is_rh_next'] #目标变量
    
    print("Feature的矩阵规模：", features.shape)
    print("Tag的矩阵规模：", labels.shape)
    
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3, random_state=0)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("训练集Tag的正样本比例：", labels_train.mean())
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print("测试集Tag的正样本比例：", labels_test.mean())
    print('Train Finished!')
    
    features_train2=features_train.drop(['ASSET_ROW_ID'],axis=1)
    features_test2=features_test.drop(['ASSET_ROW_ID'],axis=1)
    
    clf = DecisionTreeClassifier(max_depth=7,class_weight='balanced',min_samples_leaf=10,min_samples_split=60).fit(features_train2, labels_train)
    print('Train Finished!')
    plt.figure(figsize=(10, 5))
    show_score_comparation(clf, features_train2, features_test2,labels_train, labels_test)
    
    out = pd.concat([pd.DataFrame(np.array(features_test.ASSET_ROW_ID)),pd.DataFrame(clf.predict_proba(features_test2))
                     ,pd.DataFrame(np.array(labels_test))],axis=1)
    out.to_csv("./Probability_total_clf.csv",index=False,sep=',')    
        
    path=u'/ddhome/tools/yun'
    df=pd.DataFrame(pd.read_csv(os.path.join(path,'markting_datav3.csv'),encoding='utf-8'))
    df.head(10) 
    features=df.iloc[:,0:-1]
    features=features.drop(['ASSET_ROW_ID','is_kdts','is_itv_up','is_mobile_up'],axis=1)
    
    labels=df['is_rh_next'] #目标变量
    
    print("Feature的矩阵规模：", features.shape)
    print("Tag的矩阵规模：", labels.shape)    
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3, random_state=0)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("训练集Tag的正样本比例：", labels_train.mean())
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print("测试集Tag的正样本比例：", labels_test.mean())
    print('Train Finished!')
    rf = RandomForestClassifier()
    rf.fit(features_train, labels_train.ravel())
    
    plt.figure(figsize=(10,5))
    show_score_comparation(rf,features_train, features_test,labels_train, labels_test)
    plt.show()
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_jobs = -1)
                            ,param_grid ={'n_estimators':[50,100,150,200]}
                            ,scoring='roc_auc',cv=3)
    gsearch1.fit(features_train, labels_train.ravel())
    print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)    
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_jobs = -1,n_estimators=160)
                            ,param_grid ={'max_features':['auto','sqrt','log2']},scoring='roc_auc',cv=3)
    gsearch1.fit(features_train, labels_train.ravel())
    print(gsearch1.cv_results_['mean_test_score'],gsearch1.cv_results_['params'], gsearch1.best_params_, gsearch1.best_score_)   
        
    pipeline = Pipeline([
        ('rf', RandomForestClassifier(n_jobs = 1,class_weight='balanced',n_estimators=100,max_features='auto'))
    ])
    
    parameters = {
        'rf__max_depth': (9,10),
        'rf__min_samples_split': (20,25),
        'rf__min_samples_leaf': (40,45)
    }
    
    model_selection = GridSearchCV(
        pipeline, parameters, scoring='roc_auc')
    model_selection.fit(features_train, labels_train.ravel())
    print('最佳效果：%0.3f' % model_selection.best_score_)
    print('最优参数')
    best_parameters = model_selection.best_best_parameters = model_selection.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))     
    rf = RandomForestClassifier(n_jobs = -1,class_weight='balanced',n_estimators=200,max_features='sqrt'
                                ,max_depth=10,min_samples_leaf=40,min_samples_split=25)
    rf.fit(features_train, labels_train)
    
    y_pred = rf.predict(features_test)
    # features_train, features_test, labels_train, labels_test
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(labels_test, y_pred)
    np.set_printoptions(precision=2)
    
    
    print('------------混淆矩阵-------------')
    print(cnf_matrix)
    print("P值 in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[0, 1] + cnf_matrix[1, 1]))
    print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
    
    plt.figure(figsize=(10,5))
    show_score_comparation(rf,features_train, features_test,labels_train, labels_test)
    plt.show()
    
    features=df.iloc[:,0:-1]
    features=features.drop(['is_kdts','is_itv_up','is_mobile_up'],axis=1)
    labels=df['is_rh_next'] #目标变量
    
    print("Feature的矩阵规模：", features.shape)
    print("Tag的矩阵规模：", labels.shape)
    
    features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3, random_state=0)
    print("训练集Feature的矩阵规模：", features_train.shape)
    print("训练集Tag的矩阵规模：", labels_train.shape)
    print("训练集Tag的正样本比例：", labels_train.mean())
    print("测试集Feature的矩阵规模：", features_test.shape)
    print("测试集Tag的矩阵规模：", labels_test.shape)
    print("测试集Tag的正样本比例：", labels_test.mean())
    print('Train Finished!')
    
    features_train2=features_train.drop(['ASSET_ROW_ID'],axis=1)
    features_test2=features_test.drop(['ASSET_ROW_ID'],axis=1)
    
    rf = RandomForestClassifier(n_jobs = -1,class_weight='balanced',n_estimators=200,max_features='sqrt'
                                ,max_depth=10,min_samples_leaf=40,min_samples_split=25)
    rf.fit(features_train2, labels_train)
    print('Train Finished!')
    plt.figure(figsize=(10, 5))
    show_score_comparation(clf, features_train2, features_test2,labels_train, labels_test)
    
    out = pd.concat([pd.DataFrame(np.array(features_test.ASSET_ROW_ID)),pd.DataFrame(rf.predict_proba(features_test2))
                     ,pd.DataFrame(np.array(labels_test))],axis=1)
    out.to_csv("./Probability_total_rf.csv",index=False,sep=',')














