import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit

workclass = {'Federal-gov': 0, 'Local-gov': 1, 'Private': 2, 'Self-emp-inc': 3, 'Self-emp-not-inc': 4, 'State-gov': 5, 'Without-pay': 6}
education = {'10th': 0, '11th': 1, '12th': 2, '1st-4th': 3, '5th-6th': 4, '7th-8th': 5, '9th': 6, 'Assoc-acdm': 7, 'Assoc-voc': 8, 'Bachelors': 9, 'Doctorate': 10, 'HS-grad': 11, 'Masters': 12, 'Preschool': 13, 'Prof-school': 14, 'Some-college': 15}
maritalstatus = {'Divorced': 0, 'Married-AF-spouse': 1, 'Married-civ-spouse': 2, 'Married-spouse-absent': 3, 'Never-married': 4, 'Separated': 5, 'Widowed': 6}
occupation = {'Adm-clerical': 0, 'Armed-Forces': 1, 'Craft-repair': 2, 'Exec-managerial': 3, 'Farming-fishing': 4, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Other-service': 7, 'Priv-house-serv': 8, 'Prof-specialty': 9, 'Protective-serv': 10, 'Sales': 11, 'Tech-support': 12, 'Transport-moving': 13}
relationship = {'Husband': 0, 'Not-in-family': 1, 'Other-relative': 2, 'Own-child': 3, 'Unmarried': 4, 'Wife': 5}
race = {'Amer-Indian-Eskimo': 0, 'Asian-Pac-Islander': 1, 'Black': 2, 'Other': 3, 'White': 4}
sex = {'Female': 0, 'Male': 1}
nativecountry = {'Cambodia': 0, 'Canada': 1, 'China': 2, 'Columbia': 3, 'Cuba': 4, 'Dominican-Republic': 5, 'Ecuador': 6, 'El-Salvador': 7, 'England': 8, 'France': 9, 'Germany': 10, 'Greece': 11, 'Guatemala': 12, 'Haiti': 13, 'Holand-Netherlands': 14, 'Honduras': 15, 'Hong': 16, 'Hungary': 17, 'India': 18, 'Iran': 19, 'Ireland': 20, 'Italy': 21, 'Jamaica': 22, 'Japan': 23, 'Laos': 24, 'Mexico': 25, 'Nicaragua': 26, 'Outlying-US(Guam-USVI-etc)': 27, 'Peru': 28, 'Philippines': 29, 'Poland': 30, 'Portugal': 31, 'Puerto-Rico': 32, 'Scotland': 33, 'South': 34, 'Taiwan': 35, 'Thailand': 36, 'Trinadad&Tobago': 37, 'United-States': 38, 'Vietnam': 39, 'Yugoslavia': 40}
salary = {'<=50K': -1, '>50K': 1}
salary2 = {'<=50K.': -1, '>50K.': 1}

def func(x, a, b):
    return a / x + b

def plotAUC(auc, title):
    n_bl = np.linspace(1, len(auc), len(auc))
    plt.ylabel('AUC')
    plt.xlabel("Number of Base Learners")
    plt.plot(n_bl, auc)
    # popt, pcov = curve_fit(func, n_bl, auc)
    # plt.plot(n_bl, func(n_bl, popt[0], popt[1]), color='red', linestyle=':')
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.title(title)
    plt.savefig(title)
    plt.clf()

def LoadDataFrame(file):
    df = np.array(pd.read_csv(file, header=None, sep=',\s', engine = 'python'))
    df = np.delete(df, np.where(df=='?')[0], axis=0)
    df = pd.DataFrame(df)
    df[1] = df[1].map(workclass)
    df[3] = df[3].map(education)
    df[5] = df[5].map(maritalstatus)
    df[6] = df[6].map(occupation)
    df[7] = df[7].map(relationship)
    df[8] = df[8].map(race)
    df[9] = df[9].map(sex)
    df[13] = df[13].map(nativecountry)
    df[14] = df[14].map(salary)
    return df

def LoadTrainData(file):
    df = np.array(pd.read_csv(file, header=None, sep=',\s', engine = 'python'))
    df = np.delete(df, np.where(df=='?')[0], axis=0)
    df = pd.DataFrame(df)
    df[1] = df[1].map(workclass)
    df[3] = df[3].map(education)
    df[5] = df[5].map(maritalstatus)
    df[6] = df[6].map(occupation)
    df[7] = df[7].map(relationship)
    df[8] = df[8].map(race)
    df[9] = df[9].map(sex)
    df[13] = df[13].map(nativecountry)
    df[14] = df[14].map(salary)
    df = np.array(df)
    m , n = df.shape
    data = df[:,0:n-1]
    label = df[:,-1].astype('int')
    return data, label

def LoadTestData(file):
    df = np.array(pd.read_csv(file, header=None, sep=',\s', engine = 'python'))
    df = np.delete(df, np.where(df=='?')[0], axis=0)
    df = pd.DataFrame(df)
    df[1] = df[1].map(workclass)
    df[3] = df[3].map(education)
    df[5] = df[5].map(maritalstatus)
    df[6] = df[6].map(occupation)
    df[7] = df[7].map(relationship)
    df[8] = df[8].map(race)
    df[9] = df[9].map(sex)
    df[13] = df[13].map(nativecountry)
    df[14] = df[14].map(salary2)
    df = np.array(df)
    m , n = df.shape
    data = df[:,0:n-1]
    label = df[:,-1].astype('int')
    return data, label

# AdaBoost主要算法
def Train(data, label, Num_BaseLearner):
    H = []
    A = []
    D = np.zeros((len(data)))+1./len(data)
    T = Num_BaseLearner
    for i in range(T):
        h = tree.DecisionTreeClassifier(max_depth = 5)
        h = h.fit(data, label, sample_weight = D)
        pre_h = h.predict(data)
        e = np.dot(D, pre_h!=label)
        if e > 0.5:
            break
        a = 0.5 * np.log((1-e)/e)
        H.append(h)
        A.append(a)
        z = np.dot(np.exp(-a * label * pre_h), D)
        D = D * np.exp(-a * label * pre_h) / z
    return H, A

# 预测
def Predict(H, A, data):
    res = np.zeros(len(data))
    for i in range(len(H)):
        res = res + A[i]*H[i].predict(data)
    res = np.sign(res)
    return res

# 给定base learner个数Num_BaseLearner，用5折交叉验证法测试，AUC作为评估标准，返回AUC值
def TrainUsing5Fold(data, label, Num_BaseLearner):
    auc_score = []
    kf = KFold(n_splits=5, shuffle=False)
    for train_index , test_index in kf.split(data):
        H, A = Train(data[train_index], label[train_index], Num_BaseLearner)
        auc_score.append(roc_auc_score(label[test_index], Predict(H, A, data[test_index])))
    auc_score = np.array(auc_score)
    print(auc_score)
    print(np.mean(auc_score))
    return np.mean(auc_score)

# 获取不同个数的base learners所产生的AUC值
def Test(data, label, maxNum_BaseLearner):
    auc = []
    # base learner个数从1遍历到maxNum_BaseLearner
    for i in range(maxNum_BaseLearner):
        print(i+1)
        auc.append(TrainUsing5Fold(data, label, i+1))
    auc = np.array(auc)
    print(auc)
    # plot出不同个数的base learners所产生的AUC值
    plotAUC(auc, "Effect of the Number of Base Learners on AdaBoost Performance")