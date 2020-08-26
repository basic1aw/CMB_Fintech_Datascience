import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
tags = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/训练数据集/训练数据集_tag.csv')
pd.options.display.max_columns = 40
tags[tags['job_year']==99]
# 注意到job_year = 99 的这个用户 age = 36, 基于事实推断原则，认定他的年龄正确，但工作年限
# 不属实,所以可以根据教育程度和年龄来推断工作年限
tags.isnull().sum().sort_values() # atdd_type缺失值比例太大，弃之
tags['job_year'].quantile(0.99)
tags_test = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/评分数据集/评分数据集_tag.csv')
tags['crd_card_act_ind'].value_counts()
tags.groupby('atdd_type')['l1y_crd_card_csm_amt_dlm_cd'].mean()
tags['l1y_crd_card_csm_amt_dlm_cd'].value_counts() # 0-5
tags['crd_card_act_ind'].value_counts() # 0 - 1
tags[tags['cur_credit_cnt']==tags['cur_credit_cnt'].max()]

drop_list = ['cur_debit_cnt','cur_credit_cnt']
# tags = tags.drop(columns=drop_list)
# tags_test = tags_test.drop(columns=drop_list)
tags.drop(columns=['deg_cd','edu_deg_cd','atdd_type'],inplace=True)
tags_test.drop(columns=['deg_cd','edu_deg_cd','atdd_type'],inplace=True)

from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
def knn_fillna(x_train,y_train,test,k=9,dispersed=False):
    """x_train: 不含缺失值的数据（不包括目标列）
    y_train: 不含缺失值的目标列
    test: 目标列缺失值所在行的其他数据
    """
    if dispersed:
        knn = KNeighborsClassifier(n_neighbors=k)
    else:
        knn = KNeighborsRegressor(n_neighbors=k,weights='distance')
# 根据要填充的缺失值的变量类型来确定是分类还是回归预测
    knn.fit(x_train,y_train)
    return knn.predict(test)


for tag in [tags,tags_test]:
    tag.loc[:,'gdr_cd'] = tag['gdr_cd'].map({'M':1,'F':0})
    tag['mrg_situ_cd'] = tag['mrg_situ_cd'].map({'A':2,'B':1,'0':0,'O':3,'Z':4,'~':5})
    tag['acdm_deg_cd'] = tag['acdm_deg_cd'].map({'Z':4,'31':3,'G':5,'F':6,
                                                'C':0,'D':1,'30':2})
    #先转换成数值，即label encoding，然后便于用KNN填充缺失值

def fill(tag):
    notnull = []
    null = []
    if ('flag' in tag.columns):
        column = tag.drop(columns=['id','flag']).columns
    else:
        column = tag.drop(columns=['id']).columns
    for i in column:
        if tag[i].notnull().all():
            notnull.append(i)
        else:
            null.append(i)
    for i in null:
        y_notnull = tag.loc[tag[i].notnull(),i] # 用于训练的ylabel，即含有缺失值的该列的非缺失值
        idx_notnull = y_notnull.index
        idx_null = tag.loc[tag[i].isnull(),i].index   # 含缺失值的该列的缺失值对应的索引值index
        x_notnull = tag[notnull].loc[idx_notnull] # 全部都是非缺失值的列，作为xlabel来预测含缺失值的列
        x_null = tag[notnull].loc[idx_null]       # 用于测试的xlabel，即含有缺失值的该列缺失值所对应的其他全为非缺失值的列
        if i in ['gdr_cd','mrg_situ_cd','acdm_deg_cd']:
            predicted = knn_fillna(x_notnull,y_notnull,x_null,dispersed=True)
        else:
            predicted = knn_fillna(x_notnull,y_notnull,x_null,dispersed=False)
        tag.loc[idx_null,i] = predicted


for i in [tags,tags_test]:
    fill(i)
# 在进行KNN填充后对分类数据进行on-hot-encoding

for tag in [tags,tags_test]:
    tag.loc[tag['age']<=35,'age'] = 0
    tag.loc[(tag['age']>35)&(tag['age']<=51),'age'] = 1
    tag.loc[(tag['age']>51)&(tag['age']<=67),'age'] = 2
    tag.loc[(tag['age']>67),'age'] = 3
    tag.loc[(tag['cur_debit_cnt']>0),'cur_debit_cnt'] = 1
    tag.loc[(tag['cur_credit_cnt']>0),'cur_credit_cnt'] = 1
    tag.loc[:,'debit'] = tag['cur_debit_cnt'] * tag['cur_debit_min_opn_dt_cnt']
    tag.loc[:,'credit'] = tag['cur_credit_cnt'] * tag['cur_credit_min_opn_dt_cnt']
    tag.loc[tag['cur_debit_crd_lvl']==0,'cur_debit_crd_lvl'] = 2
    tag.loc[(tag['cur_debit_crd_lvl']>0)&(tag['cur_debit_crd_lvl']<=10),
            'cur_debit_crd_lvl'] = 3
    tag.loc[(tag['cur_debit_crd_lvl']>10)&(tag['cur_debit_crd_lvl']<=20),
            'cur_debit_crd_lvl'] = 1
    tag.loc[tag['cur_debit_crd_lvl']>20,'cur_debit_crd_lvl'] = 0
    tag.loc[tag['perm_crd_lmt_cd']<=0,'perm_crd_lmt_cd'] = 0
    tag.loc[(tag['perm_crd_lmt_cd']>0)&(tag['perm_crd_lmt_cd']<=3),'perm_crd_lmt_cd'] = 1
    tag.loc[(tag['perm_crd_lmt_cd']>3)&(tag['perm_crd_lmt_cd']<7),'perm_crd_lmt_cd'] = 3
    tag.loc[tag['perm_crd_lmt_cd']>=7,'perm_crd_lmt_cd'] = 2
    tag.loc[(tag['l6mon_daim_aum_cd']>=0)&(tag['l6mon_daim_aum_cd']<3),
            'l6mon_daim_aum_cd'] = 2
    tag.loc[tag['l6mon_daim_aum_cd']>=3,'l6mon_daim_aum_cd'] = 0
    tag.loc[tag['l6mon_daim_aum_cd']<=-1,'l6mon_daim_aum_cd'] = 1
    tag.loc[(tag['bk1_cur_year_mon_avg_agn_amt_cd']>=0)&(tag['bk1_cur_year_mon_avg_agn_amt_cd']<5),
            'bk1_cur_year_mon_avg_agn_amt_cd'] = 1
    tag.loc[(tag['bk1_cur_year_mon_avg_agn_amt_cd']>=5),
            'bk1_cur_year_mon_avg_agn_amt_cd'] = 2
    tag.loc[tag['bk1_cur_year_mon_avg_agn_amt_cd']==-1,
            'bk1_cur_year_mon_avg_agn_amt_cd'] = 0
    tag.loc[(tag['pl_crd_lmt_cd']==0),'pl_crd_lmt_cd'] = 2
    tag.loc[(tag['pl_crd_lmt_cd']==-1),'pl_crd_lmt_cd'] = 3
    tag.loc[(tag['pl_crd_lmt_cd']>0)&(tag['pl_crd_lmt_cd']<5),
            'pl_crd_lmt_cd'] = 1
    tag.loc[tag['pl_crd_lmt_cd']>5,'pl_crd_lmt_cd'] = 0
    tag.loc[tag['debit']<3000,'debit'] = 2
    tag.loc[(tag['debit']>=3000)&(tag['debit']<5000),'debit'] = 1
    tag.loc[tag['debit']>=5000,'debit'] = 0
    tag.loc[tag['credit']<2029,'credit'] = 2
    tag.loc[(tag['credit']>=2029)&(tag['credit']<4058),'credit'] = 1
    tag.loc[tag['credit']>=4058,'credit'] = 0

j99 = np.zeros((3,4))
for i in range(3):
    for j in range(4):
        k = tags[(tags['bk1_cur_year_mon_avg_agn_amt_cd']==i)&(tags['age']==j)]['job_year'].dropna().mean()
        j99[i,j] = k
for i in range(3):
    for j in range(4):
        tags.loc[(tags['job_year']==99)&(tags['bk1_cur_year_mon_avg_agn_amt_cd']==i)&(tags['age']==j),
                'job_year'] = j99[i,j]

for tag in [tags,tags_test]:
    tag.loc[:,'gdr_cd'] = tag['gdr_cd'].map({1:'M',0:'F'})

    tag['mrg_situ_cd'] = tag['mrg_situ_cd'].map({2:'MA',1:'MB',0:'M0',3:'MO',4:'MZ',5:'MX'})

    tag['acdm_deg_cd'] = tag['acdm_deg_cd'].map({4:'DEG_Z',3:'DEG_31',5:'DEG_G',6:'DEG_F',
                                                 0:'DEG_C',1:'DEG_D',2:'DEG_30'})

for col in ['gdr_cd','mrg_situ_cd','acdm_deg_cd']:
    tags = tags.join(pd.get_dummies(tags[col]))
    tags.drop(columns=col,inplace=True)
    tags_test = tags_test.join(pd.get_dummies(tags_test[col]))
    tags_test.drop(columns=col,inplace=True)


X_train = tags.drop(['id','flag'],axis=1).values
Y_train = tags['flag'].values
X_test = tags_test.drop(['id'],axis=1).values


beh_test = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/评分数据集/评分数据集_beh.csv')
beh_train = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/训练数据集/训练数据集_beh.csv')
trd_train = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/训练数据集/训练数据集_trd.csv')
trd_test = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/评分数据集/评分数据集_trd.csv')
beh_train[beh_train['flag']==1]['page_no'].value_counts()
trd_train[trd_train['flag']==0]['cny_trx_amt'].mean() # 发现违约的人群中交易均值为负数，即支出
# 为违约的人群交易均值为正数，即收入
trd_train.groupby('Dat_Flg1_Cd')['flag'].mean()
# 仅根据交易方向判断是不够的，还要看总的交易额
trd_train['trx_tm'].astype('datetime64')

# p_group = pd.DataFrame(beh_train.groupby('id')['page_no'].apply(np.unique))
# # np.unique来对以根据id进行分组的page_no 去重，以此实现one-hot-encoding
#
# for page in page_set:
#     p_group.loc[:,page] = page
#     def check(label):
#         return int(i in label)
#     p_group.loc[:,i] = p_group.loc[:,'page_no'].apply(check)
# p_group.drop(columns='page_no',inplace=True)
page_set = beh_train.groupby('page_no')['flag'].mean().sort_values(ascending=False).index.values
page_set = np.sort(page_set)
p_train = pd.DataFrame(beh_train.groupby('id')['page_no'].apply(np.unique))
p_test = pd.DataFrame(beh_test.groupby('id')['page_no'].apply(np.unique))

for p in [p_train,p_test]:
    for page in page_set:
            p.loc[:,page] = page # 把所有页面都合并到DF中
            def check(label):
                return int(page in label)
                # 逐个检查每个用户是否访问过该页面
            p.loc[:,page] = p.loc[:,'page_no'].apply(check)
    p.drop(columns='page_no',inplace=True)


# trd_train[trd_train['cny_trx_amt']<0]['Dat_Flg1_Cd'].value_counts()
# trd_train.shape
# len(trd_train['id'].value_counts().index) # 查看id有多少unique value
# trd_train['Dat_Flg3_Cd'].value_counts()
# pit = pd.pivot_table(data=trd_train,values='flag',
#                index=['Dat_Flg1_Cd','Trx_Cod1_Cd','Trx_Cod2_Cd'],
#                columns='Dat_Flg3_Cd',aggfunc=np.mean)

tre_train = pd.DataFrame(trd_train.groupby('id')['cny_trx_amt'].mean())
tre_train.loc[:,'trx_std'] = trd_train.groupby('id')['cny_trx_amt'].std()
tre_test = pd.DataFrame(trd_test.groupby('id')['cny_trx_amt'].mean())
tre_test.loc[:,'trx_std'] = trd_test.groupby('id')['cny_trx_amt'].std()
for i in [tre_train,tre_test]:
    i.loc[(i['cny_trx_amt'].notnull())&(i['trx_std'].isnull()),'trx_std'] = 0
tre_train.isnull().sum()
df_train = tags.join(tre_train,on='id')
df_test = tags_test.join(tre_test,on='id')
train = df_train.join(p_train,on='id')
test = df_test.join(p_test,on='id')

def fillnull(df,label1,label2,target,avg=True):
    n1 = len(df[label1].value_counts().index)
    n2 = len(df[label2].value_counts().index)
    tempt = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            assume = df[(df[label1]==i)&(df[label2]==j)][target].dropna()
            if avg == True:
                tempt[i,j] = assume.mean()
            else:
                tempt[i,j] = assume.median()
    for i in range(n1):
        for j in range(n2):
            df.loc[(df[target].isnull())&(df[label1]==i)&(df[label2]==j),
                    target] = round(tempt[i,j])


def fill_page(tag):
    """用来填充page，因为都转成了one-hot，所以按照分类处理
    """
    notnull = []
    null = []
    if ('flag' in tag.columns):
        column = tag.drop(columns=['id','flag']).columns
    else:
        column = tag.drop(columns=['id']).columns
    for i in column:
        if tag[i].notnull().all():
            notnull.append(i)
        else:
            null.append(i)
    for i in null:
        y_notnull = tag.loc[tag[i].notnull(),i] # 用于训练的ylabel，即含有缺失值的该列的非缺失值
        idx_notnull = y_notnull.index
        idx_null = tag.loc[tag[i].isnull(),i].index   # 含缺失值的该列的缺失值对应的索引值index
        x_notnull = tag[notnull].loc[idx_notnull] # 全部都是非缺失值的列，作为xlabel来预测含缺失值的列
        x_null = tag[notnull].loc[idx_null]       # 用于测试的xlabel，即含有缺失值的该列缺失值所对应的其他全为非缺失值的列
        predicted = knn_fillna(x_notnull,y_notnull,x_null,dispersed=True)
        tag.loc[idx_null,i] = predicted

def normalize(number):
    """ Z_score, a way of normalizing data
    """
    return (number-np.mean(number))/np.std(number)

for m in [train,test]:
    fillnull(m,'l6mon_daim_aum_cd','age','trx_std')
    fillnull(m,'l6mon_daim_aum_cd','bk1_cur_year_mon_avg_agn_amt_cd',
          'cny_trx_amt')
    fill_page(m)
    # 用KNN填充page中的缺失值


for i in [train,test]:
    for j in ['cny_trx_amt','trx_std']:
        i[j] = normalize(i[j])
    i.loc[i['l12mon_buy_fin_mng_whl_tms']>0,'has_bt_fin'] = 1
    i.loc[i['l12mon_buy_fin_mng_whl_tms']==0,'has_bt_fin'] = 0
    i.loc[i['l12_mon_fnd_buy_whl_tms']>0,'has_bt_whl'] = 1
    i.loc[i['l12_mon_fnd_buy_whl_tms']==0,'has_bt_whl'] = 0
    i.loc[i['l12_mon_insu_buy_whl_tms']>0,'has_bt_insu'] = 1
    i.loc[i['l12_mon_insu_buy_whl_tms']==0,'has_bt_insu'] = 0
    i.loc[i['l12_mon_gld_buy_whl_tms']>0,'has_bt_gld'] = 1
    i.loc[i['l12_mon_gld_buy_whl_tms']==0,'has_bt_gld'] = 0
    i.drop(columns=['l12_mon_gld_buy_whl_tms','l12_mon_insu_buy_whl_tms','l12_mon_fnd_buy_whl_tms',
                   'l12mon_buy_fin_mng_whl_tms'],inplace=True)
# 仅仅查看是否有购买过理财产品

for i in [train,test]:
    i.drop(columns=drop_list,inplace=True)
    i.loc[i['job_year']<=14,'job_year'] = 0
    i.loc[(i['job_year']>14)&(i['job_year']<=28),'job_year'] = 1
    i.loc[i['job_year']>28,'job_year'] = 2
    for col in train.drop(columns='id').columns:
        if train[col].max() > 100:
            train[col] = normalize(train[col])
        i.loc[:,col] = i.loc[:,col].apply(round).astype('int')

# drop frs_agn_dt_cnt','his_lng_ovd_day'后本地auc略有上升，线上下降
train.info()
test.info()
# x_train = train.drop(['id','flag'],axis=1).values
# y_train = train['flag'].values
# x_test = test.drop('id',axis=1).values

# from sklearn import neighbors
# from sklearn.model_selection import cross_val_score
# from sklearn import metrics
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier,plot_importance
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
#
# lr = LogisticRegression(solver='lbfgs',max_iter=2000)
# auc_lr = cross_val_score(lr,x_train,y_train,scoring='roc_auc',cv=6).mean()
# lr.fit(x_train,y_train)
# pred_lr = lr.predict_proba(x_test)[:,1]
#
# from sklearn.svm import SVC
# svc = SVC(gamma='auto',max_iter=1000)
# auc_svc = cross_val_score(svc,x_train,y_train,scoring='roc_auc',cv=6).mean()
# n_range = range(70,120,10)
# auc_tag = []
# auc_1 = []
# for n in n_range:
#     rfc = RandomForestClassifier(n_estimators=n)
#     score1 = cross_val_score(rfc,x_train,y_train,scoring='roc_auc',
#                             cv=6).mean()
# #     score2 = cross_val_score(rfc,X_train,Y_train,scoring='roc_auc',cv=6).mean()
#     auc_1.append(score1)
#     auc_tag.append(score2)
# plt.plot(n_range,auc_tag,marker='.',label='with only tag')
# plt.plot(n_range,auc_1,marker='.',label='with 3 tables')
# plt.xlabel('number of estimators')
# plt.ylabel('AUC score')
# plt.legend()
# plt.show()
# n_estimators = 130
# rfc = RandomForestClassifier(n_estimators=110)
# auc_rfc = cross_val_score(rfc,x_train,y_train,scoring='roc_auc',
#                           cv=6).mean()
# auc_rfc

# rfc = RandomForestClassifier(n_estimators=110)
# rfc.fit(x_train,y_train)
# pred_rfc = rfc.predict_proba(x_test)[:,1]
# submission_rfc = pd.DataFrame({'id':test['id'].values,'flag_predicted':pred_lr})
# sub_rfc = submission_rfc.sort_values(by='id')
# path_rfc = 'X:/CS2/招商银行Fintech/SUB/LR/LR_split_page_drop_loan.csv'
# sub_rfc.to_csv(path_rfc,header=False,index=False)
# best score 0.67 for n_estimators = 100, 0.6 for cross_val_score
# add flag into x_label when using knn filling null, got best n_estimators=310,0.612 for cv


# knn = KNeighborsClassifier(n_neighbors=15)
# auc_knn = cross_val_score(knn,x_train,y_train,cv=6,scoring='roc_auc').mean()
# knn.fit(x_train,y_train)
# pred_knn = knn.predict_proba(x_test)[:,1]
# submission_rfc = pd.DataFrame({'id':test['id'].values,'flag_predicted':pred_knn})
# sub_rfc = submission_rfc.sort_values(by='id')
# path_rfc = 'X:/CS2/招商银行Fintech/SUB/KNN/submission_knn_fill_23.csv'
# sub_rfc.to_csv(path_rfc,header=False,index=False)



# xgb = XGBClassifier()
#
# auc_xgb = cross_val_score(xgb,x_train,y_train,cv=6,scoring='roc_auc').mean()
# auc_xgb
# print('rfc={},knn={},xgb={},LR={},SVC={}'.format(auc_rfc,auc_knn,auc_xgb,auc_lr,auc_svc))
# xgb.fit(x_train,y_train)
# pred_xgb = xgb.predict_proba(x_test)[:,1]
# submission_rfc = pd.DataFrame({'id':test['id'].values,'flag_predicted':pred_xgb})
# sub_rfc = submission_rfc.sort_values(by='id')
# path_rfc = 'X:/CS2/招商银行Fintech/SUB/xgboost/xgb_knn_split_page_drop_card.csv'
# sub_rfc.to_csv(path_rfc,header=False,index=False)
# # cross_val_score=0.627, cv=0.628 after adding flag into knn fill
# plot_importance(xgb,height=0.5,max_num_features=60)
