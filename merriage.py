#匯入模組
from numpy.lib.function_base import i0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 

df = pd.read_csv("merriage2.csv")
df.describe()
df.info()
df.head()

#清洗資料
#將教育程度改成「大學以上」和「高中以下」兩種，因為就讀碩博士的人數較少，會影響判斷
#並且將「大學以上」設成 1 ; 「高中以下」設成 0
educa = df['edu'].values
print(educa)
count = 0
high = []
univer = []
for i in educa:
    if i == '博士畢業' or i == '碩士畢業' or i == '大學畢業' or i == '專科畢業':
        univer.append(count)
        count += 1       
    else:
        high.append(count)
        count += 1
u = len(univer)
h = len(high)
for i in univer:
    df.loc[i,'edu'] = int(1)
for j in high:
    df.loc[j,'edu'] = int(0)
print(educa)
df.head()

#未滿15歲～24歲的結婚人數不多，所以合併成「24歲以下」
age = df['age'].values
countage = 0
cantmarry = []
for i in age:
    if i == '未滿15歲' or i == '15 ~ 19 歲' or i == '20 ~ 24 歲':
        cantmarry.append(countage)
    countage += 1
    
count_del = 0
for n in cantmarry:
    df.loc[n,'age'] = '24歲以下'
print(age)

#檢查資料
df[1:20]
df[-2:]
df[-1:]
#丟掉最後一筆NaN的資料
df.drop(df.index[310800], inplace=True)
df[310799:]
#利用散佈圖探討關係程度
#年齡與離婚人數（結論：相關度很高）
plt.xlabel('age')
plt.ylabel('divorce_count')
plt.scatter(df["age"], df["divorce_count"], c='m',s=50 ,alpha = 0.2)
plt.show()
#教育程度與離婚人數（結論：相關度高，但是可能和國民教育水平有關）
plt.xlabel('edu')
plt.ylabel('divorce_count')
plt.scatter(df["edu"], df["divorce_count"], c='m',s=50 ,alpha = 0.2)
plt.show()
#地區與離婚人數（結論：新北和台北的離婚人數多，但是可能和居住人口數有關）
plt.xlabel('site_id')
plt.ylabel('divorce_count')
plt.scatter(df["site_id"], df["divorce_count"], c='m',s=50 ,alpha = 0.2)
plt.show()
#性別與離婚人數（結論：相關度不高）
plt.xlabel(' sex')
plt.ylabel('divorce_count')
plt.scatter(df[" sex"], df["divorce_count"], c='m',s=50 ,alpha = 0.2)
plt.show()

#決定取相關度最高的年齡和教育程度
#將年齡和教育程度以數值代替
#24以下=1, 25~29=2, 30~34=3, 35~39=4, 40~44=5, 45~49=6, 50~54=7, 55~59=8, 60~64=9, 65以上=10
age = df['age'].values
count_age = 0
A=[]
B=[]
C=[]
D=[]
E=[]
F=[]
G=[]
H=[]
I=[]
J=[]
for i in age:
    if i == '24歲以下':
        A.append(count_age)
        count_age += 1
    elif i == '25 ~ 29 歲':
        B.append(count_age) 
        count_age += 1   
    elif i == '30 ~ 34 歲':
        C.append(count_age) 
        count_age += 1   
    elif i == '35 ~ 39 歲':
        D.append(count_age)
        count_age += 1 
    elif i == '40 ~ 44 歲':
        E.append(count_age) 
        count_age += 1  
    elif i == '45 ~ 49 歲':
        F.append(count_age) 
        count_age += 1
    elif i == '50 ~ 54 歲':
        G.append(count_age) 
        count_age += 1
    elif i == '55 ~ 59 歲':
        H.append(count_age) 
        count_age += 1
    elif i == '60 ~ 64 歲':
        I.append(count_age) 
        count_age += 1
    else:
        J.append(count_age)    
        count_age += 1

u = len(univer)
h = len(high)
for i in A:
    df.loc[i,'age'] = int(1)
for i in B:
    df.loc[i,'age'] = int(2)
for i in C:
    df.loc[i,'age'] = int(3)
for i in D:
    df.loc[i,'age'] = int(4)
for i in E:
    df.loc[i,'age'] = int(5)
for i in F:
    df.loc[i,'age'] = int(6)
for i in G:
    df.loc[i,'age'] = int(7)
for i in H:
    df.loc[i,'age'] = int(8)
for i in I:
    df.loc[i,'age'] = int(9)
for i in J:
    df.loc[i,'age'] = int(10)

#看離婚人數的分布狀況（結論：0佔據大部分）
# （結論：離婚人數0=極低機率、1~2=低機率、3~5=高機率、>6=超高機率）
dc = df['divorce_count'].values
print(np.percentile(dc,[25,50,75]))
print(np.percentile(dc,[75,85,95])) 
print(np.percentile(dc,[90,94,98]))


#資料分割/建模
#邏輯回歸
from sklearn import preprocessing, linear_model
X = pd.DataFrame([df['age'],df['edu']]).T
y= df['divorce_count']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=900)
logistic = linear_model.LogisticRegression()

logistic.fit(X,y)

print("迴歸係數：", logistic.coef_[100])
print('截距：',logistic.intercept_[100])

#迴歸模型準確度
preds = logistic.predict(X)
print(pd.crosstab(preds, df['divorce_count']))
print(logistic.score(X,y))




#決策樹
from sklearn import linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn import preprocessing
import pydotplus
from IPython.display import Image

df[33390:33490]

#建模(演算法：決策樹)
y = df['divorce_count']
X = df.drop(["site_id",' sex','nation','divorce_count'],axis=1)
X.info()

tree_model = DecisionTreeClassifier(random_state=0, max_depth=5)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=99) 
tree_model.fit(X_train, y_train)
predictions = tree_model.predict(X_test) 
predictions

#準確率
tree_model.score(X_test, y_test)

#交叉分析表
preds = tree_model.predict_proba(X=X_test)
print(pd.crosstab(preds[:,0], columns=[X_test["age"],X_test["edu"]]))


#數據結論
'''
1. age的數字意義：24以下=1, 25~29=2, 30~34=3, 35~39=4, 40~44=5, 45~49=6, 50~54=7, 55~59=8, 60~64=9, 65以上=10
2. edu的數字意義：將「大學以上」設成 1 ; 「高中以下」設成 0
3. 決策樹的機率代表「不會離婚的機率」eg.0.983604較不可能離婚，0.652440較有可能離婚
'''

#函式(介面用)
def divorceChance(age,edu):
    divorce_rate = 0
    r = ''
    if edu == 0:
        if age>=3 & age<=6:
            divorce_rate = (1-0.652440)
        elif age == 1:
            divorce_rate  = (1-0.903414)
        elif age == 2:
            divorce_rate  = (1-0.69484)
        elif age == 7:
            divorce_rate  = (1-0.728309)
        elif age == 8:
            divorce_rate  = (1-0.761504)
        elif age == 9:
            divorce_rate  = (1-0.806492)
        elif age == 10:
            divorce_rate  = (1-0.842117)
    
    if edu==1:
        if age == 1:
            divorce_rate = (1-0.983604)
        elif age == 2:
            divorce_rate = (1-0.865665)
        elif age >= 3 & age<=5:
            divorce_rate = (1-0.812245)
        elif age == 6:
            divorce_rate = (1-0.852427)
        elif age == 7:
            divorce_rate = (1-0.873273)
        elif age == 8:
            divorce_rate = (1-0.910380)
        elif age == 9:
            divorce_rate = (1-0.934623)
        elif age == 10:
            divorce_rate = (1-0.948144)

    if divorce_rate <= (1-0.728309):
        r = '高機率離婚'
    elif divorce_rate <= (1-0.812245):
        r = '中高機率離婚'
    elif divorce_rate <= (1-0.865665):
        r = '中低機率離婚'
    elif divorce_rate <= (1-0.910380):
        r = '低機率離婚'
    elif divorce_rate <= (1-0.983604):
        r = '極低機率離婚'
    

    return r
    