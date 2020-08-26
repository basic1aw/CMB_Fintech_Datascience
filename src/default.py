import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
tags = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/训练数据集/训练数据集_tag.csv')
pd.options.display.max_columns = 40
tags.head()
tags['id'].is_unique # tag 表中用户 id unique

def show(column,bins=10):
    plt.hist(column,label=column.name,bins=bins)
    plt.legend()
    plt.show()

tags.loc[:,'color'] = tags['flag'].map({1:'red',0:'grey'}).values

def barplot(column):
    default = tags[tags['flag'] == 1][column].value_counts()
    undefault = tags[tags['flag'] == 0][column].value_counts()
    grouped = pd.DataFrame({'default':default,'undefualt':undefault})
    grouped.plot(kind='bar',stacked=True)
    plt.xlabel(column)
    plt.show()

def grouped_ratio(column):
    """ return the ratio of defaulters in each group split by given column
    """
    return tags.groupby(column)['flag'].mean()

tags[tags['job_year']==99]
# 注意到job_year = 99 的这个用户 age = 36, 基于事实推断原则，认定他的年龄正确，但工作年限
# 不属实,所以可以根据教育程度和年龄来推断工作年限
tags.isnull().sum().sort_values() # atdd_type缺失值比例太大，弃之
tags['job_year'].quantile(0.99)
def scatterplot(column1,column2):
    axc = tags.plot(x=column1,y=column2,kind='scatter',color=tags['color'],s=8)
    axc.set_xlim([0,25])
    axc.set_ylim([0,25])
tags_test = pd.read_csv('X:/CS2/招商银行Fintech/A榜赛题/评分数据集/评分数据集_tag.csv')
tags['crd_card_act_ind'].value_counts()
tags.groupby('atdd_type')['l1y_crd_card_csm_amt_dlm_cd'].mean()
tags['l1y_crd_card_csm_amt_dlm_cd'].value_counts() # 0-5
tags['crd_card_act_ind'].value_counts() # 0 - 1
tags[tags['cur_debit_cnt']==178]
tags[tags['cur_credit_cnt']==tags['cur_credit_cnt'].max()]
# scatterplot('cur_debit_cnt','cur_credit_cnt')

def normalize(number):
    """ Z_score, a way of normalizing data
    """
    return (number-np.mean(number))/np.std(number)

def r(df,x,y):
    """return the correlation efficiency of dataset x and y in df
    """
    return np.mean(normalize(df[x]) * normalize(df[y]))

# tags.loc[:,'card'] = tags['cur_debit_cnt'] + tags['cur_credit_cnt']
# credit and flag has higher r
# tags.isnull().sum().sort_values()
drop_list = ['tot_ast_lvl_cd','cust_inv_rsk_endu_lvl_cd','frs_agn_dt_cnt',
             'dnl_bind_cmb_lif_ind','dnl_mbl_bnk_ind','ic_ind','pot_ast_lvl_cd',
             'deg_cd','edu_deg_cd','vld_rsk_ases_ind','l12_mon_gld_buy_whl_tms',
             'l12_mon_insu_buy_whl_tms','l12_mon_fnd_buy_whl_tms',
            'l12mon_buy_fin_mng_whl_tms','atdd_type','cur_debit_min_opn_dt_cnt','cur_credit_min_opn_dt_cnt',
                     'cur_debit_cnt','cur_credit_cnt']
tags.groupby('age')['flag'].mean()

# 注意到持有借记卡天数与持有借记卡数量可以合并为一个feature，
# 先把数量get_dummy为0，1然后再与天数相乘，信用卡同理
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
    tag.loc[:,'gdr_cd'] = tag['gdr_cd'].map({'M':1,'F':0})
    tag['mrg_situ_cd'] = tag['mrg_situ_cd'].map({'A':2,'B':1,'0':0,'O':3,'Z':4,'~':np.nan})
    tag['acdm_deg_cd'] = tag['acdm_deg_cd'].map({'Z':2,'31':2,'G':3,'F':3,
                                                'C':0,'D':0,'30':1})


(tags['cur_debit_crd_lvl'] == 0).sum(),(tags['cur_debit_cnt']==0).sum()
# cur_debit_crd_lvl中0表示没有持有借记卡
(tags['hld_crd_card_grd_cd'] == -1).sum(),(tags['cur_credit_cnt']==0).sum()
# tags['hld_crd_card_grd_cd'].value_counts()
# 而信用卡中等级为-1的数量为2709，同时没有持有信用卡的人数为2633,由此可知-1表示
# 包括没有信用卡
notnull = []
null = []
for i in tags.columns:
    if tags[i].notnull().all():
        notnull.append(i)
    else:
        null.append(i)
def knn_fillna(x_train,y_train,test,k=3,dispersed=True):
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
for i in null:
    x = tags[null].values
    y = tags.loc[tags[i].notnull(),i].values
    to_be_filled = tags.loc[tags[i].isnull(),i].values
    predicted = knn_fillna(x,y,to_be_filled,k=3)
    tags.loc[to_be_filled.index,i] = predicted
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
j99 = np.zeros((3,4))
for i in range(3):
    for j in range(4):
        k = tags[(tags['bk1_cur_year_mon_avg_agn_amt_cd']==i)&(tags['age']==j)]['job_year'].dropna().mean()
        j99[i,j] = k
for i in range(3):
    for j in range(4):
        tags.loc[(tags['job_year']==99)&(tags['bk1_cur_year_mon_avg_agn_amt_cd']==i)&(tags['age']==j),
                'job_year'] = j99[i,j]


for i in [tags,tags_test]:
    fillnull(i,'bk1_cur_year_mon_avg_agn_amt_cd','age','acdm_deg_cd')
    fillnull(i,'pl_crd_lmt_cd','bk1_cur_year_mon_avg_agn_amt_cd','fin_rsk_ases_grd_cd')
    fillnull(i,'pl_crd_lmt_cd','bk1_cur_year_mon_avg_agn_amt_cd','confirm_rsk_ases_lvl_typ_cd')
    fillnull(i,'perm_crd_lmt_cd','age','job_year')
    fillnull(i,'perm_crd_lmt_cd','pl_crd_lmt_cd','hld_crd_card_grd_cd',False)
    fillnull(i,'perm_crd_lmt_cd','pl_crd_lmt_cd','l1y_crd_card_csm_amt_dlm_cd')
    # 信用卡额度与消费分层和信用卡等级，信用等有关
    fillnull(i,'perm_crd_lmt_cd','age','gdr_cd')
    i.loc[(i['crd_card_act_ind'].isnull())&(i['l1y_crd_card_csm_amt_dlm_cd']==0),'crd_card_act_ind'] = 0
    i.loc[(i['crd_card_act_ind'].isnull())&(i['l1y_crd_card_csm_amt_dlm_cd']>0),'crd_card_act_ind'] = 1
    # 最近一年有消费则表示活跃
    fillnull(i,'l6mon_daim_aum_cd','pl_crd_lmt_cd','fr_or_sh_ind')
    fillnull(i,'debit','job_year','l6mon_agn_ind')
    fillnull(i,'age','pl_crd_lmt_cd','hav_car_grp_ind')
    fillnull(i,'bk1_cur_year_mon_avg_agn_amt_cd','pl_crd_lmt_cd','hav_hou_grp_ind')
    fillnull(i,'cur_debit_min_opn_dt_cnt','bk1_cur_year_mon_avg_agn_amt_cd','l6mon_agn_ind')
    i.loc[(i['fin_rsk_ases_grd_cd']>=0)&(i['fin_rsk_ases_grd_cd']<3),
          'fin_rsk_ases_grd_cd'] = 1
    i.loc[i['fin_rsk_ases_grd_cd']>=3,'fin_rsk_ases_grd_cd'] = 2
    i.loc[i['fin_rsk_ases_grd_cd']<0,'fin_rsk_ases_grd_cd'] = 0
    i.loc[i['ovd_30d_loan_tot_cnt']>=1,'ovd_30d_loan_tot_cnt'] = 1
    fillnull(i,'pl_crd_lmt_cd','perm_crd_lmt_cd','his_lng_ovd_day')
    i.loc[:,'has_overdue'] = (i['his_lng_ovd_day']>0).astype('int')
    fillnull(i,'fin_rsk_ases_grd_cd','pl_crd_lmt_cd','loan_act_ind')
    fillnull(i,'loan_act_ind','pl_crd_lmt_cd','ovd_30d_loan_tot_cnt')
    fillnull(i,'age','gdr_cd','mrg_situ_cd')
    i.loc[i['confirm_rsk_ases_lvl_typ_cd']<1,'confirm_rsk_ases_lvl_typ_cd'] = 0
    i.loc[(i['confirm_rsk_ases_lvl_typ_cd']>=1)&(i['confirm_rsk_ases_lvl_typ_cd']<4),
       'confirm_rsk_ases_lvl_typ_cd'] = 1
    i.loc[i['confirm_rsk_ases_lvl_typ_cd']>=4,'confirm_rsk_ases_lvl_typ_cd'] = 2
    i.loc[i['job_year']<1,'job_year'] = 3
    i.loc[(i['job_year']>=1)&(i['job_year']<10),'job_year'] = 2
    i.loc[(i['job_year']>=10)&i['job_year']<25,'job_year'] = 1
    i.loc[i['job_year']>=25] = 0
    i.loc[i['hld_crd_card_grd_cd']>=30,'hld_crd_card_grd_cd'] = 2
    i.loc[(i['hld_crd_card_grd_cd']>=10)&(i['hld_crd_card_grd_cd']<30),'hld_crd_card_grd_cd'] = 3
    i.loc[i['hld_crd_card_grd_cd']<0,'hld_crd_card_grd_cd'] = 1
tags = tags.drop(columns='his_lng_ovd_day')
tags_test = tags_test.drop(columns='his_lng_ovd_day')
tags = tags.drop(columns=drop_list)
tags_test = tags_test.drop(columns=drop_list)
tags = tags.drop(columns=['color'])
for i in [tags,tags_test]:
    for column in i.drop('id',axis=1).columns:
        i[column] = i[column].astype('int')


X_train = tags.drop(['id','flag'],axis=1).values
Y_train = tags['flag'].values
X_test = tags_test.drop(['id'],axis=1).values
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
rft = RandomForestClassifier(n_estimators=178)
rft.fit(X_train,Y_train)
pred_rft = rft.predict_proba(X_test)[:,1]
# estimators_range = range(50,201)
# auc_random = []
# for n in estimators_range:
#     auc_random.append(auc_rft(n,X_train,Y_train))

# plt.plot(estimators_range,auc_random)
# plt.xlabel('number of estimators')
# plt.ylabel('AUC')
# optimal n_estimators = 178

"""
without considering other two tables (beh,trx)
"""
# for other two tables
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
l3 = page[:5]
l2 = page[5:16]
l1 = page[16:]
beh = [beh_test,beh_train]
for i in beh:
    for pe in l3:
        i.loc[i['page_no']==pe,'page_no'] = 2
    for pe in l2:
        i.loc[i['page_no']==pe,'page_no'] = 1
    for pe in l1:
        i.loc[i['page_no']==pe,'page_no'] = 0
    i['page_no'] = i['page_no'].astype('int')


trd_train[trd_train['cny_trx_amt']<0]['Dat_Flg1_Cd'].value_counts()
trd_train.shape
len(trd_train['id'].value_counts().index) # 查看id有多少unique value
trd_train['Dat_Flg3_Cd'].value_counts()
pit = pd.pivot_table(data=trd_train,values='flag',
               index=['Dat_Flg1_Cd','Trx_Cod1_Cd','Trx_Cod2_Cd'],
               columns='Dat_Flg3_Cd',aggfunc=np.mean)

# trd_train[trd_train['Dat_Flg1_Cd']=='C']['Trx_Cod1_Cd'].value_counts()
# 发现支出中支付方式只有1，3，收入中支付方式只有2，3
# pit.loc[('C',3,),:].mean(axis=1).mean()
# pit.loc[('C',2,),:].mean(axis=1).mean()
# money = pd.pivot_table(data=trd_train,values='cny_trx_amt',
#                index=['Dat_Flg1_Cd','Dat_Flg3_Cd','Trx_Cod1_Cd'],
#                columns='flag',aggfunc=np.mean)
# trd_test.isnull().sum()
# 原始表格都是非空的
tre_train = pd.DataFrame(trd_train.groupby('id')['cny_trx_amt'].mean())
tre_train.loc[:,'trx_std'] = trd_train.groupby('id')['cny_trx_amt'].std()
tre_test = pd.DataFrame(trd_test.groupby('id')['cny_trx_amt'].mean())
tre_test.loc[:,'trx_std'] = trd_test.groupby('id')['cny_trx_amt'].std()
for i in [tre_train,tre_test]:
    i.loc[(i['cny_trx_amt'].notnull())&(i['trx_std'].isnull()),'trx_std'] = 0
tre_train.isnull().sum()
df_train = tags.join(tre_train,on='id')
df_test = tags_test.join(tre_test,on='id')
page_train = beh_train.groupby('id')['page_no'].mean()
page_test = beh_test.groupby('id')['page_no'].mean()
train = df_train.join(page_train,on='id')
test = df_test.join(page_test,on='id')
# for i in [train,test]:
#     i.loc[(i['cny_trx_amt'].notnull())&(i['trx_std'].isnull()),'trx_std'] = 0
# amt非空而trx空值的填充为0

train.drop('page_no',inplace=True,axis=1)
test.drop('page_no',inplace=True,axis=1)
# 舍弃page

for i in [train,test]:
    fillnull(i,'l6mon_daim_aum_cd','bk1_cur_year_mon_avg_agn_amt_cd',
          'cny_trx_amt')
    q1,q2,q3 = train['cny_trx_amt'].quantile([0.25,0.5,0.75])
    i.loc[i['cny_trx_amt']<q1,'cny_trx_amt'] = 0
    i.loc[(i['cny_trx_amt']<q2)&(i['cny_trx_amt']>=q1),'cny_trx_amt'] = 1
    i.loc[(i['cny_trx_amt']<q3)&(i['cny_trx_amt']>=q2),'cny_trx_amt'] = 2
    i.loc[(i['cny_trx_amt']>=q3),'cny_trx_amt'] = 3
    i['cny_trx_amt'] = i['cny_trx_amt'].astype('int')

    fillnull(i,'l1y_crd_card_csm_amt_dlm_cd','l6mon_daim_aum_cd',
          'trx_std')
    m1,m2,m3 = train['trx_std'].quantile([0.25,0.5,0.75])
    i.loc[i['trx_std']<round(m1),'trx_std'] = 0
    i.loc[(i['trx_std']>=round(m1))&(i['trx_std']<round(m2)),'trx_std'] = 1
    i.loc[(i['trx_std']>=round(m2))&(i['trx_std']<round(m3)),'trx_std'] = 2
    i.loc[(i['trx_std']>=round(m3)),'trx_std'] = 3
    i['trx_std'] = i['trx_std'].fillna(0).astype('int')

test['trx_std'].isnull().sum()
# import seaborn as sns
# sns.boxplot(train['cny_trx_amt'])
# train['trx_std'].isnull().sum()
# test['trx_std'].isnull().sum()
# for i in [train,test]:

x_train = train.drop(['id','flag'],axis=1).values
y_train = train['flag'].values
x_test = test.drop('id',axis=1).values
# n_range = range(50,200,10)
# auc_tag = []
# auc_1 = []
# for n in n_range:
#     rfc = RandomForestClassifier(n_estimators=n)
#     score1 = cross_val_score(rfc,x_train,y_train,scoring='roc_auc',
#                             cv=6).mean()
#     score2 = cross_val_score(rfc,X_train,Y_train,scoring='roc_auc',cv=6).mean()
#     auc_1.append(score1)
#     auc_tag.append(score2)
# plt.plot(n_range,auc_tag,marker='.',label='with only tag')
# plt.plot(n_range,auc_1,marker='.',label='with 3 tables')
# plt.xlabel('number of estimators')
# plt.ylabel('AUC score')
# plt.legend()
# plt.show()
# n_estimators = 70,with x_train,auc=0.61 for cv=6

# rfc = RandomForestClassifier(n_estimators=70)
# rfc.fit(x_train,y_train)
# rfc.score(x_train,y_train)
# auc_d_tag = []
# auc_tag = []
# auc_d_3 = []
# auc_3 = []
# k_range = range(1,50,5)
# for k in k_range:
#     clf = neighbors.KNeighborsClassifier(n_neighbors=k)
#     clf_d = neighbors.KNeighborsClassifier(n_neighbors=k,weights='distance')
#     score1 = cross_val_score(clf,X_train,Y_train,cv=6,scoring='roc_auc').mean()
#     score2 = cross_val_score(clf,x_train,y_train,cv=6,scoring='roc_auc').mean()
#     score3 = cross_val_score(clf_d,X_train,Y_train,cv=6,scoring='roc_auc').mean()
#     score4 = cross_val_score(clf_d,x_train,y_train,cv=6,scoring='roc_auc').mean()
#     auc_tag.append(score1)
#     auc_d_tag.append(score3)
#     auc_3.append(score2)
#     auc_d_3.append(score4)
# plt.plot(k_range,auc_tag,marker='.',label='tag-no-distance')
# plt.plot(k_range,auc_d_tag,marker='.',label='tag-with-distance')
# plt.plot(k_range,auc_3,marker='.',label='3labels-no-distance')
# plt.plot(k_range,auc_d_3,marker='.',label='3labels-with-distance')
# plt.xlabel('number of k_neighbors')
# plt.ylabel('AUC')
# plt.legend()
# plt.show()
# k= 48,with 3tabels,no distance

# auc_final = []
# k_range = range(35,101,5)
# for k in k_range:
#     clf = neighbors.KNeighborsClassifier(n_neighbors=k)
#     score1 = cross_val_score(clf,x_train,y_train,cv=6,scoring='roc_auc').mean()
#     auc_final.append(score1)
# plt.plot(k_range,auc_final,marker='.',label='3tables-no-distance')
# plt.xlabel('number of k_neighbors')
# plt.ylabel('AUC')
# plt.legend()
# plt.show()
# check k_range(40,81),find a max at k=75,auc=0.6765
# decide k = 75, for large k,泛化能力更好

# rfc = RandomForestClassifier(n_estimators=70)
# rfc.fit(x_train,y_train)
# pred_rfc = rfc.predict_proba(x_test)[:,1]
# submission_rfc = pd.DataFrame({'id':test['id'].values,'flag_predicted':pred_rfc})
# sub_rfc = submission_rfc.sort_values(by='id')
# path_rfc = 'X:/CS2/招商银行Fintech/SUB/random_forest/submission_n_70.csv'
# sub_rfc.to_csv(path_rfc)
# sub_rfc.head()
# knn = neighbors.KNeighborsClassifier(n_neighbors=95)
# knn.fit(x_train,y_train)
# pred_knn = rfc.predict_proba(x_test)[:,1]
# submission_knn = pd.DataFrame({'id':test['id'].values,'flag':pred_knn})
# path_knn = 'X:/CS2/招商银行Fintech/SUB/KNN/submission_k_95.csv'
# sub_knn = submission_knn.sort_values('id')
# sub_knn.to_csv(path_knn,header=False,index=False)
