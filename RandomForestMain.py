import AdaBoost
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# 放回采样
def getsample(dataframe):
    df = dataframe.sample(n=len(dataframe), replace=True, axis=0)
    df = np.array(df)
    m , n = df.shape
    data = df[:,0:n-1]
    label = df[:,-1].astype('int')
    return data, label

# Random Forest主要算法
def RFTrain(dataframe, Num_BaseLearner):
    H = []
    for i in range(Num_BaseLearner):
        train_data_sample, train_label_sample = getsample(dataframe)
        h = tree.DecisionTreeClassifier(max_depth=5, splitter='best', max_features='log2')
        h = h.fit(train_data_sample, train_label_sample)
        H.append(h)
    return H

# 预测
def RFPredict(H, data):
    res = np.zeros(len(data))
    for i in range(len(H)):
        res = res + H[i].predict(data)
    res = np.sign(res)
    return res

# 给定base learner个数Num_BaseLearner，用5折交叉验证法测试，AUC作为评估标准，返回AUC值
def RFTrainUsing5Fold(dataframe, Num_BaseLearner):
    auc_score = []
    kf = KFold(n_splits=5, shuffle=False)
    dataframe = np.array(dataframe)
    for train_index, test_index in kf.split(dataframe):
        df = dataframe[train_index]
        H = RFTrain(pd.DataFrame(df), Num_BaseLearner)
        df = dataframe[test_index]
        m , n = df.shape
        data = df[:,0:n-1]
        label = df[:,-1].astype('int')
        auc_score.append(roc_auc_score(label, RFPredict(H, data)))
    auc_score = np.array(auc_score)
    print(auc_score)
    print(np.mean(auc_score))
    return np.mean(auc_score)

# 获取不同个数的base learners所产生的AUC值
def RFTest(dataframe, maxNum_BaseLearner):
    auc = []
    # base learner个数从1遍历到maxNum_BaseLearner
    for i in range(maxNum_BaseLearner):
        print(i+1)
        auc.append(RFTrainUsing5Fold(dataframe,i+1))
    auc = np.array(auc)
    print(auc)
    # plot出不同个数的base learners所产生的AUC值
    AdaBoost.plotAUC(auc, "Effect of the Number of Base Learners on Random Forest Performance")


maxNum = 200                # 测试过程中，基学习器个数选择的最大值
numofbaselearner_ab = 100   # AdaBoost算法最优的基学习器个数
numofbaselearner_rf = 60    # RandomForest算法最优的基学习器个数

if __name__ == "__main__":

    # 导入数据
    train_data, train_label = AdaBoost.LoadTrainData("adult.data")
    df = AdaBoost.LoadDataFrame("adult.data")
    test_data, test_label = AdaBoost.LoadTestData("adult.test")

    '''
    # 测试不同的基学习器个数：1~maxNum
    AdaBoost.Test(train_data, train_label, maxNum)
    RFTest(df, maxNum)
    '''
    
    #  用最优的基学习器个数训练
    H_adaboost, A_adaboost = AdaBoost.Train(train_data, train_label, numofbaselearner_ab)
    H_randomforest = RFTrain(df, numofbaselearner_rf)

    # 测试并打印AUC值
    print("AUC of AdaBoost Algorithm with %d base learners: " % (numofbaselearner_ab))
    print(roc_auc_score(test_label, AdaBoost.Predict(H_adaboost, A_adaboost, test_data)))
    print("AUC of Random Forest Algorithm with %d base learners: " % (numofbaselearner_rf))
    print(roc_auc_score(test_label, RFPredict(H_randomforest, test_data)))