import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
tags = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/训练数据集/训练数据集_tag.csv')
pd.options.display.max_columns = 40
tags.head()
tags['id'].is_unique # tag 表中用户 id unique

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

drop_list = ['cur_debit_cnt','cur_credit_cnt','frs_agn_dt_cnt']
tags.groupby('age')['flag'].mean()
# tags = tags.drop(columns=drop_list)
# tags_test = tags_test.drop(columns=drop_list)
tags.drop(columns=['deg_cd','edu_deg_cd','atdd_type'],inplace=True)
tags_test.drop(columns=['deg_cd','edu_deg_cd','atdd_type'],inplace=True)
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
def knn_fillna(x_train,y_train,test,k=11,dispersed=True):
    """x_train: 不含缺失值的数据（不包括目标列）
    y_train: 不含缺失值的目标列
    test: 目标列缺失值所在行的其他数据
    """
    if dispersed:
        knn = KNeighborsClassifier(n_neighbors=k,weights='distance')
    else:
        knn = KNeighborsRegressor(n_neighbors=k,weights='distance')

    knn.fit(x_train,y_train)
    return knn.predict(test)
for tag in [tags,tags_test]:
    tag.loc[:,'gdr_cd'] = tag['gdr_cd'].map({'M':1,'F':0})
    tag['mrg_situ_cd'] = tag['mrg_situ_cd'].map({'A':2,'B':1,'0':0,'O':3,'Z':4,'~':5})
    tag['acdm_deg_cd'] = tag['acdm_deg_cd'].map({'Z':4,'31':3,'G':5,'F':6,
                                                'C':0,'D':1,'30':2})

# notnull = []
# null = []
# for i in tags.drop(columns=['id','flag']).columns:
#     if tags[i].notnull().all():
#         notnull.append(i)
#     else:
#         null.append(i)

# for i in null:
#     y_notnull = tags.loc[tags[i].notnull(),i] # 用于训练的ylabel，即含有缺失值的该列的非缺失值
#     idx_notnull = y_notnull.index
#     idx_null = tags.loc[tags[i].isnull(),i].index   # 含缺失值的该列的缺失值对应的索引值index
#     x_notnull = tags[notnull].loc[idx_notnull] # 全部都是非缺失值的列，作为xlabel来预测含缺失值的列
#     x_null = tags[notnull].loc[idx_null]       # 用于测试的xlabel，即含有缺失值的该列缺失值所对应的其他全为非缺失值的列
#     predicted = knn_fillna(x_notnull,y_notnull,x_null,k=3)
#     tags.loc[idx_null,i] = predicted


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
        predicted = knn_fillna(x_notnull,y_notnull,x_null)
        tag.loc[idx_null,i] = predicted

for i in [tags,tags_test]:
    fill(i)

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
            'bk1_cur_year_mon_avg_agn_amt_cd'] = 0
    tag.loc[tag['bk1_cur_year_mon_avg_agn_amt_cd']==-1,
            'bk1_cur_year_mon_avg_agn_amt_cd'] = 2
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
X_train = tags.drop(['id','flag'],axis=1).values
Y_train = tags['flag'].values
X_test = tags_test.drop(['id'],axis=1).values

beh_test = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/评分数据集/评分数据集_beh.csv')
beh_train = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/训练数据集/训练数据集_beh.csv')
trd_train = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/训练数据集/训练数据集_trd.csv')
trd_test = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/评分数据集/评分数据集_trd.csv')
beh_train[beh_train['flag']==1]['page_no'].value_counts()
trd_train.head()
trd_train[trd_train['flag']==0]['cny_trx_amt'].mean() # 发现违约的人群中交易均值为负数，即支出
# 为违约的人群交易均值为正数，即收入
trd_train.groupby('Dat_Flg1_Cd')['flag'].mean()
# 仅根据交易方向判断是不够的，还要看总的交易额
trd_train.head()
trd_train['trx_tm'].astype('datetime64')
page = beh_train.groupby('page_no')['flag'].mean().sort_values(ascending=False).index.values
# key_page = []
# for i in  page:
#     if i in (beh_train.loc[beh_train['flag']==1,'page_no'].values):
#         if i not in (beh_train.loc[beh_train['flag']==0,'page_no'].values):
#             key_page.append(i)
# key_page
group = beh_train.groupby('flag')['page_no'].value_counts()
# 访问量前几的且flag=1，0都有的说明并非关键page，很可能是登录页面账户页面之类的
flag_0 = pd.DataFrame({'times':group.loc[0].values},index=group.loc[0].index)
flag_1 = pd.DataFrame({'times':group.loc[1].values},index=group.loc[1].index)
compare = flag_0.join(flag_1,rsuffix='_flag=1',lsuffix='_flag=0')
compare.loc[:,'times_flag=0'] = compare['times_flag=0']/sum(compare['times_flag=0'])
compare.loc[:,'times_flag=1'] = compare['times_flag=1']/sum(compare['times_flag=1'])
# 换成比例更好比较,如果某些页面flag=1访问的比例明显高于flag=0，说明可能与default相关
# print(list(page).index('TRN'))
compare.loc[:,'diff'] = compare['times_flag=1'] - compare['times_flag=0']

# key_page = ['CQC','CQD']
# for page in compare.index:
#     if compare.loc[page,'diff'] >= 0.001:
#         key_page.append(page)
key_page = page[:14]
# compare
fg = pd.DataFrame(beh_train.groupby('id')['flag'].mean())
for page in key_page:
    beh_train.loc[:,page] = (beh_train.loc[:,'page_no'] == page).astype('int')


for i in key_page:
    s = beh_train.groupby('id')[i].sum()
    fg = fg.join(s) # fg.join!!

# # A/B test
# # null hypothesis: page will not affect flag.
# # alternative hypothesis: flag=1 will visit page more frequently,or page less frequently

def a_b_test(label,alfa,repetitions=1000):
    """ return if label has effect on flag at the given significance
    """
    diffs = np.array([])
    group = fg.groupby('flag')[label].mean()
    ob_diff = (group.loc[1] - group.loc[0])
    for i in range(repetitions):
        fg.loc[:,'shuffled'] = fg.sample(n=fg.shape[0])[label].values
        group_shuffled = fg.groupby('flag')['shuffled'].mean()
        statistic = group_shuffled.loc[1] - group_shuffled.loc[0]
        diffs = np.append(diffs,statistic)
    empirical = pd.DataFrame({'diffs':diffs})
    right = empirical['diffs'].quantile(1-alfa*0.5)
    left = empirical['diffs'].quantile(alfa*0.5)
#     ax = empirical.plot(kind='hist')
#     ax.scatter(x=ob_diff,y=0,s=30,color='violet',label=label)
#     ax.plot((left,right),(0,0),label='confidence interval for {}'.format(1-alfa),lw=3)
#     plt.legend()
#     plt.show()
    return ((ob_diff>right) or (ob_diff<left))


# significant = dict()
# for i in key_page:
#     significant[i] = a_b_test(i,0.05,3000)
# 最终选择 AAO,CQA,XAI,SZA,EGA 作为key_page

key_page = ['AAO','CQA','XAI','SZA','EGA']
for i in [beh_train,beh_test]:
    for j in key_page:
        i.loc[:,j] = (i.loc[:,'page_no'] == j).astype('int')


page_train = pd.DataFrame(beh_train.groupby('id')['AAO'].sum())
page_test = pd.DataFrame(beh_test.groupby('id')['AAO'].sum())
for i in key_page[1:]:
    page_train = page_train.join(beh_train.groupby('id')[i].sum())
    page_test = page_test.join(beh_test.groupby('id')[i].sum())


trd_train[trd_train['cny_trx_amt']<0]['Dat_Flg1_Cd'].value_counts()
trd_train.shape
len(trd_train['id'].value_counts().index) # 查看id有多少unique value
trd_train['Dat_Flg3_Cd'].value_counts()
pit = pd.pivot_table(data=trd_train,values='flag',
               index=['Dat_Flg1_Cd','Trx_Cod1_Cd','Trx_Cod2_Cd'],
               columns='Dat_Flg3_Cd',aggfunc=np.mean)
tre_train = pd.DataFrame(trd_train.groupby('id')['cny_trx_amt'].mean())
tre_train.loc[:,'trx_std'] = trd_train.groupby('id')['cny_trx_amt'].std()
tre_test = pd.DataFrame(trd_test.groupby('id')['cny_trx_amt'].mean())
tre_test.loc[:,'trx_std'] = trd_test.groupby('id')['cny_trx_amt'].std()
for i in [tre_train,tre_test]:
    i.loc[(i['cny_trx_amt'].notnull())&(i['trx_std'].isnull()),'trx_std'] = 0
tre_train.isnull().sum()
df_train = tags.join(tre_train,on='id')
df_test = tags_test.join(tre_test,on='id')
train = df_train.join(page_train,on='id')
test = df_test.join(page_test,on='id')

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


def fill_single(tag,label,discrete=False):
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

    y_notnull = tag.loc[tag[label].notnull(),label] # 用于训练的ylabel，即含有缺失值的该列的非缺失值
    idx_notnull = y_notnull.index
    idx_null = (tag.loc[tag[label].isnull(),label]).index   # 含缺失值的该列的缺失值对应的索引值index
    x_notnull = tag[notnull].loc[idx_notnull] # 全部都是非缺失值的列，作为xlabel来预测含缺失值的列
    x_null = tag[notnull].loc[idx_null]       # 用于测试的xlabel，即含有缺失值的该列缺失值所对应的其他全为非缺失值的列
    predicted = knn_fillna(x_notnull,y_notnull.astype('int'),x_null,k=5)
    tag.loc[idx_null,i] = predicted

def normalize(number):
    """ Z_score, a way of normalizing data
    """
    return (number-np.mean(number))/np.std(number)
for m in [train,test]:
    for n in key_page:
        m.loc[m['dnl_mbl_bnk_ind']==0,n] = 0
        i[n] = normalize(m[n])
        fillnull(m,'l1y_crd_card_csm_amt_dlm_cd','l6mon_daim_aum_cd',n)
    fillnull(m,'l1y_crd_card_csm_amt_dlm_cd','l6mon_daim_aum_cd',
          'trx_std')
    fillnull(m,'l6mon_daim_aum_cd','bk1_cur_year_mon_avg_agn_amt_cd',
          'cny_trx_amt')




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
# drop2 = [','frs_agn_dt_cnt','his_lng_ovd_day'']
for i in [train,test]:
    i.drop(columns=drop_list,inplace=True)
    for j in train.drop(columns=['id']).columns:
        if train[j].max() > 100:
            train[j] = normalize(train[j])
    i['job_year'] = normalize(i['job_year'])

# drop frs_agn_dt_cnt','his_lng_ovd_day'后本地auc略有上升，线上下降
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

# lr = LogisticRegression(solver='lbfgs',max_iter=2000)
# auc_lr = cross_val_score(lr,x_train,y_train,scoring='roc_auc',cv=6).mean()
# lr.fit(x_train,y_train)
# pred_lr = lr.predict_proba(x_test)[:,1]

# from sklearn.svm import SVC
# svc = SVC(gamma='auto',max_iter=1000)
# auc_svc = cross_val_score(svc,x_train,y_train,scoring='roc_auc',cv=6).mean()
# n_range = range(70,120,10)
# # auc_tag = []
# # auc_1 = []
# # for n in n_range:
# #     rfc = RandomForestClassifier(n_estimators=n)
# #     score1 = cross_val_score(rfc,x_train,y_train,scoring='roc_auc',
# #                             cv=6).mean()
# # #     score2 = cross_val_score(rfc,X_train,Y_train,scoring='roc_auc',cv=6).mean()
# #     auc_1.append(score1)
# #     auc_tag.append(score2)
# # plt.plot(n_range,auc_tag,marker='.',label='with only tag')
# # plt.plot(n_range,auc_1,marker='.',label='with 3 tables')
# # plt.xlabel('number of estimators')
# # plt.ylabel('AUC score')
# # plt.legend()
# # plt.show()
# # n_estimators = 130
# # rfc = RandomForestClassifier(n_estimators=110)
# # auc_rfc = cross_val_score(rfc,x_train,y_train,scoring='roc_auc',
# #                           cv=6).mean()
# # auc_rfc

# # rfc = RandomForestClassifier(n_estimators=110)
# # rfc.fit(x_train,y_train)
# # pred_rfc = rfc.predict_proba(x_test)[:,1]
# submission_rfc = pd.DataFrame({'id':test['id'].values,'flag_predicted':pred_lr})
# sub_rfc = submission_rfc.sort_values(by='id')
# path_rfc = 'X:/CS2/招商银行Fintech/SUB/LR/LR_split_page_drop_loan.csv'
# sub_rfc.to_csv(path_rfc,header=False,index=False)
# # best score 0.67 for n_estimators = 100, 0.6 for cross_val_score
# # add flag into x_label when using knn filling null, got best n_estimators=310,0.612 for cv


# knn = KNeighborsClassifier(n_neighbors=15)
# auc_knn = cross_val_score(knn,x_train,y_train,cv=6,scoring='roc_auc').mean()
# # knn.fit(x_train,y_train)
# # pred_knn = knn.predict_proba(x_test)[:,1]
# # submission_rfc = pd.DataFrame({'id':test['id'].values,'flag_predicted':pred_knn})
# # sub_rfc = submission_rfc.sort_values(by='id')
# # path_rfc = 'X:/CS2/招商银行Fintech/SUB/KNN/submission_knn_fill_23.csv'
# # sub_rfc.to_csv(path_rfc,header=False,index=False)



# xgb = XGBClassifier()

# auc_xgb = cross_val_score(xgb,x_train,y_train,cv=6,scoring='roc_auc').mean()
# auc_xgb
# print('rfc={},knn={},xgb={},LR={},SVC={}'.format(auc_rfc,auc_knn,auc_xgb,auc_lr,auc_svc))
# # xgb.fit(x_train,y_train)
# pred_xgb = xgb.predict_proba(x_test)[:,1]
# submission_rfc = pd.DataFrame({'id':test['id'].values,'flag_predicted':pred_xgb})
# sub_rfc = submission_rfc.sort_values(by='id')
# path_rfc = 'X:/CS2/招商银行Fintech/SUB/xgboost/xgb_knn_split_page_drop_card.csv'
# sub_rfc.to_csv(path_rfc,header=False,index=False)
# # cross_val_score=0.627, cv=0.628 after adding flag into knn fill
# plot_importance(xgb,height=0.5,max_num_features=60)
