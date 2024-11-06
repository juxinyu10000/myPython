# 导入应用库文件
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno
import warnings

warnings.filterwarnings("ignore")

# 原始数据集太大，选取部分数据进行分析
loan_data_or = pd.read_csv("/bigdata/practise/riskmanagement/loan.csv").iloc[:230658]

del_precent = 0.4  # 若是全量数据分析，取0值
from sklearn.model_selection import train_test_split


def data_prepration(x):
    x_features = x.ix[:, x.columns != "loan_status"]  # 采用熟悉特征作为记录属性特征
    x_labels = x.ix[:, x.columns == "loan_status"]  # 采用贷款状态作为类别
	# 调用train_test_split拆分数据
    x_features_train, x_features_test, x_labels_train, x_labels_test = train_test_split(x_features, x_labels,
                                                                                        test_size=del_precent)  # ****抽样比例可以自己设置
    return (x_features_train, x_features_test, x_labels_train, x_labels_test)


data_A_X, data_B_X, data_A_y, data_B_y = data_prepration(loan_data_or)
loan_data = pd.concat([data_A_X, data_A_y], axis=1)

# *********************由于数据集过大，因此这里使用相关算法，对数据集进行切分****结束
thresh_count = len(loan_data) * 0.9  # 设定阀值
loan_data = loan_data.dropna(thresh=thresh_count, axis=1)  # 若某一列数据缺失的数量超过阀值就会被删除
loan_data.to_csv("./loan2007_2012.csv", index=False)


# 下面的部分需要并发测试
def concurrency_test():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import missingno as msno
    import warnings

    loans = pd.read_csv("./loan2007_2012.csv")

    # 对行进行分析
    loans = loans.loc[:, loans.apply(pd.Series.nunique) != 1]

    # select_dtypes(include=["object"])表示选择对象分类属性
    objectColumns = loans.select_dtypes(include=["object"]).columns

    loans["int_rate"] = loans["int_rate"].astype(str).astype("float")  # 类型转换
    loans["revol_util"] = loans["revol_util"].astype(str).astype("float")  # 类型转换
    objectColumns = loans.select_dtypes(include=["object"]).columns  # 选取对象类型数据
    msno.matrix(loans[objectColumns])  # 缺失值可视化
    plt.show()

    # 添加如下代码，填充缺失值。
    objectColumns = loans.select_dtypes(include=["object"]).columns
    loans[objectColumns] = loans[objectColumns].fillna("Unknown")
    numColumns = loans.select_dtypes(include=[np.number]).columns  # 选取数值变量的列
    pd.set_option("display.max_columns", len(numColumns))

    from sklearn import preprocessing
    from sklearn.impute import SimpleImputer
    imr = SimpleImputer(missing_values=np.nan, strategy="mean")
    imr = imr.fit(loans[numColumns])
    loans[numColumns] = imr.transform(loans[numColumns])
    msno.matrix(loans[numColumns])  # 缺失值可视化
    plt.show()

    # 设置过滤属性列
    drop_list = ['member_id', 'term', 'sub_grade', 'emp_title', 'issue_d', 'title', 'zip_code', 'addr_state',
                 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d', 'url', 'pymnt_plan', 'initial_list_status']
    loans.drop(drop_list, axis=1, inplace=True)  # 过滤属性
    loans.select_dtypes(include=["object"]).shape  # 输出过滤完成后的数据集维度
    loans["installment_feat"] = loans["installment"] / ((loans["annual_inc"] + 1) / 12)

    def coding(col, codeDict):  # 数据转码
        colCoded = pd.Series(col, copy=True)
        for key, value in codeDict.items():
            colCoded.replace(key, value, inplace=True)
        return colCoded

    # 把贷款状态LoanStatus编码为违约=1, 正常=0:
    loans["loan_status"] = coding(loans["loan_status"],
                                  {"Current": 0, "Issued": 0, "Fully Paid": 0, "In Grace Period": 1,
                                   "Late (31-120 days)": 1, "Late (16-30 days)": 1, "Charged Off": 1,
                                   "Does not meet the credit policy. Status:Charged Off": 1,
                                   "Does not meet the credit policy. Status:Fully Paid": 0, "Default": 0})  # 打印状态信息
    loans.select_dtypes(include=["object"]).head()

    # 有序特征的映射
    mapping_dict = {
        "emp_length": {
            "10+ years": 10,
            "9 years": 9,
            "8 years": 8,
            "7 years": 7,
            "6 years": 6,
            "5 years": 5,
            "4 years": 4,
            "3 years": 3,
            "2 years": 2,
            "1 year": 1,
            "< 1 year": 0,
            "Unknown": 0
        },
        "grade": {
            "A": 1,
            "B": 2,
            "C": 3,
            "D": 4,
            "E": 5,
            "F": 6,
            "G": 7
        }
    }

    loans = loans.replace(mapping_dict)
    n_columns = ["home_ownership", "verification_status", "application_type", "purpose"]
    dummy_df = pd.get_dummies(loans[n_columns])  # 用get_dummies进行one hot编码
    loans = pd.concat([loans, dummy_df], axis=1)
    # 当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并

    loans = loans.drop(n_columns, axis=1)
    col = loans.select_dtypes(include=["int64", "float64"]).columns  # 挑选数值属性
    col = col.drop("loan_status")  # 剔除目标变量
    loans_ml_df = loans  # 复制数据至变量loans_ml_df

    # 采用的是标准化的方法，调用scikit-learn模块preprocessing的子模块StandardScaler。
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()  # 初始化缩放器
    loans_ml_df[col] = sc.fit_transform(loans_ml_df[col])  # 对数据进行标准

    x_feature = list(loans_ml_df.columns)
    x_feature.remove("loan_status")
    x_val = loans_ml_df[x_feature]
    y_val = loans_ml_df["loan_status"]

    # from sklearn.linear_model.logistic import LogisticRegression
    from sklearn.linear_model import LogisticRegression
    # 建立逻辑回归分类器
    model = LogisticRegression()
    # 建立递归特征消除筛选器
    from sklearn.feature_selection import RFE
    rfe = RFE(model, 30)  # 通过递归选择特征，选择30个特征
    rfe = rfe.fit(x_val, y_val)
    # 打印筛选结果
    print(rfe.n_features_)
    print(rfe.estimator_)
    print(rfe.support_)
    print(rfe.ranking_)  # ranking 为 1代表被选中，其他则未被代表未被选中

    col_filter = x_val.columns[rfe.support_]
    colormap = plt.cm.viridis
    import seaborn as sns
    plt.figure(figsize=(12, 12))
    plt.title("Pearson Correlation of Features", y=1.05, size=15)
    sns.heatmap(loans_ml_df[col_filter].corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor="white",
                annot=True)
    plt.show()

    drop_col = ['funded_amnt', 'funded_amnt_inv', 'installment', 'out_prncp_inv', 'total_pymnt_inv', 'total_rec_prncp',
                'total_rec_int', 'verification_status_Not Verified', 'verification_status_Source Verified',
                'collection_recovery_fee', 'verification_status_Verified', 'application_type_JOINT']
    col_new = col_filter.drop(drop_col)  # 删除属性列表
    plt.figure(figsize=(12, 12))
    plt.title("Pearson Correlation of Features", y=1.05, size=15)
    sns.heatmap(loans_ml_df[col_new].corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor="white",
                annot=True)
    plt.show()

    from sklearn.ensemble import RandomForestClassifier
    names = loans_ml_df[col_new].columns
    clf = RandomForestClassifier(n_estimators=10, random_state=123)  # 构建分类随机森林分类器
    clf.fit(x_val[col_new], y_val)  # 对自变量和因变量进行拟合
    plt.style.use("ggplot")

    ## feature importances 可视化##
    importances = clf.feature_importances_
    feat_names = names
    indices = np.argsort(importances)[::-1]
    fig = plt.figure(figsize=(20, 6))
    plt.title("Feature importances by RandomTreeClassifier")
    plt.bar(range(len(indices)), importances[indices], color="lightblue", align="center")
    plt.step(range(len(indices)), np.cumsum(importances[indices]), where="mid", label="Cumulative")
    plt.xticks(range(len(indices)), feat_names[indices], rotation="vertical", fontsize=14)
    plt.xlim([-1, len(indices)])
    plt.show()

    # 原数据拆分，70%用于训练，30%用于测试。
    from sklearn.model_selection import train_test_split
    def data_prepration(x):
        x_features = x.ix[:, x.columns != "loan_status"]  # 采用熟悉特征作为记录属性特征
        x_labels = x.ix[:, x.columns == "loan_status"]  # 采用贷款状态作为类别
        # 调用train_test_split拆分数据
        x_features_train, x_features_test, x_labels_train, x_labels_test = train_test_split(x_features, x_labels,
                                                                                            test_size=0.3)
        return (x_features_train, x_features_test, x_labels_train, x_labels_test)

    df = loans_ml_df
    data_train_X, data_test_X, data_train_y, data_test_y = data_prepration(df)

    # 调用SMOTE算法进行倾斜数据的平衡化处理
    from imblearn.over_sampling import SMOTE
    os = SMOTE(random_state=0)
    os_data_X, os_data_y = os.fit_sample(data_train_X.values, data_train_y.values.ravel())
    columns = data_train_X.columns
    os_data_X = pd.DataFrame(data=os_data_X, columns=columns)
    os_data_y = pd.DataFrame(data=os_data_y, columns=["loan_status"])
    newtraindata = pd.concat([os_data_X, os_data_y], axis=1)
    newtestdata = pd.concat([data_test_X, data_test_y], axis=1)
    # 将处理后的数据保存到制定目录下 ***这里注意使用自己的用户目录
    newtraindata.to_csv("./train.csv", sep=",", index=False, header=False)
    newtestdata.to_csv("./test.csv", sep=",", index=False, header=False)
