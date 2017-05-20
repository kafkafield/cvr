# coding: utf-8
get_ipython().magic(u'run init.py')
get_ipython().magic(u'pylab')
train=pd.read_csv('data/train_19.csv')
get_ipython().magic(u'pylab inline')
train.head()
index=train.columns
index
index2=[u'label', u'clickTime', u'conversionTime', u'creativeID', u'userID',                   │0,170000,,1004,2038823,977,1,1,411,564,3,465,1,209,0,1,0,0,0,0,0,0,1001,17.0,0.0,0.0,1,69.0,7.
       u'positionID', u'connectionType', u'telecomsOperator', u'adID',                        │0
       u'campaignID', u'advertiserID', u'appID', u'appPlatform',                              │0,170000,,1887,2015141,3688,1,1,369,144,84,360,1,201,0,1,35,1,2,2,0,1001,1001,17.0,0.0,10.0,1,
       u'appcategory', u'sitesetID', u'positionType', u'age', u'gender',                      │0.0,1.0
       u'education', u'marriageStatus', u'haveBaby', u'hometown', u'residence',               │0,170000,,3293,1177829,3347,1,1,2891,685,80,14,2,2,0,1,30,2,1,2,1,107,107,17.0,0.0,1.0,2,0.0,0
       u'clickDay', u'clickTimeInDay', u'provinceOfHowntown', u'clickUserCnt',                          │.0
       u'appUserPastCnt', u'appUserNewCnt']
index2=[u'label', u'clickTime', u'conversionTime', u'creativeID', u'userID',                 
       u'positionID', u'connectionType', u'telecomsOperator', u'adID',                
       u'campaignID', u'advertiserID', u'appID', u'appPlatform',                      
       u'appcategory', u'sitesetID', u'positionType', u'age', u'gender',              
       u'education', u'marriageStatus', u'haveBaby', u'hometown', u'residence',       
       u'clickDay', u'clickTimeInDay', u'provinceOfHowntown', u'clickUserCnt',                  
       u'appUserPastCnt', u'appUserNewCnt']
index2
train.columns=index2
train.head
train.head()
index2=[u'label', u'clickTime', u'conversionTime', u'creativeID', u'userID',                 
       u'positionID', u'connectionType', u'telecomsOperator', u'adID',                
       u'campaignID', u'advertiserID', u'appID', u'appPlatform',                      
       u'appcategory', u'sitesetID', u'positionType', u'age', u'gender',              
       u'education', u'marriageStatus', u'haveBaby', u'hometown', u'residence',       
       u'clickDay', u'clickTimeInDay', u'provinceOfHowntown', u'clickOfUserCnt',                  
       u'appUserPastCnt', u'appUserNewCnt']
train.columns=index2
train.head()
index2=[u'label', u'clickTime', u'conversionTime', u'creativeID', u'userID',                 
       u'positionID', u'connectionType', u'telecomsOperator', u'adID',                
       u'campaignID', u'advertiserID', u'appID', u'appPlatform',                      
       u'appcategory', u'sitesetID', u'positionType', u'age', u'gender',              
       u'education', u'marriageStatus', u'haveBaby', u'hometown', u'residence',       
       u'clickDay', u'clickTimeInDay', u'provinceOfHometown', u'clickOfUserCnt',                  
       u'appUserPastCnt', u'appUserNewCnt']
train.columns=index2
train.head()
def get_provinceOfResidence(data):
    data['provinceOfResidence'] = np.floor(data['residence'] / 100)
    return data
train=get_provinceOfResidence(train)
train.head
train.head()
sns.barplot(x='provinceOfResidence',y='label',data=train)
train['clickDay'].count()
train['clickDay', 'label'].groupBy('clickDay').count()
train.groupBy('clickDay').count()
train.groupby('clickDay').count()
train['clickDay'].groupby('clickDay').count()
train['clickDay','label'].groupby('clickDay').count()
train['label','clickDay'].groupby('clickDay').count()
train.head()
train.groupby('clickDay').count()['label']
clickdaycnt=pd.DataFrame(train.groupby('clickDay').count()['label'])
clickdaycnt
clickdaycnt=clickdaycnt.rename(columns={'label':'clickofDayCnt'})
clickdaycnt
clickdaycnt=clickdaycnt.rename(columns={'clickofDayCnt':'clickOfDayCnt'})
clickdaycnt
clickdaycnt.to_csv('data/clickDayCnt.csv')
train.join(clickdaycnt, on = 'clickDay')
train=train.join(clickdaycnt, on = 'clickDay')
train.head()
clickDayCntTest=pd.DataFrame([31,100], columns={'clickDay','clickOfDayCnt'})
clickDayCntTest=pd.DataFrame([[31],[100]], columns={'clickDay','clickOfDayCnt'})
clickDayCntTest=pd.DataFrame([[31,32],[100,101]], columns={'clickDay','clickOfDayCnt'})
clickDayCntTest
clickDayCntTest=pd.DataFrame([[31,100]], columns={'clickDay','clickOfDayCnt'})
clickDayCntTest
train.head
train.head()
train.columns
train=train[              u'label',           u'clickTime',      u'conversionTime',            
                u'creativeID',              u'userID',          u'positionID',      
            u'connectionType',    u'telecomsOperator',                u'adID',      
                u'campaignID',        u'advertiserID',               u'appID',      
               u'appPlatform',         u'appcategory',           u'sitesetID',      
              u'positionType',                 u'age',              u'gender',      
                 u'education',      u'marriageStatus',            u'haveBaby',      
                  u'hometown',           u'residence',            u'clickDay',      
            u'clickTimeInDay',  u'provinceOfHometown',      u'provinceOfResidence', u'clickOfUserCnt',      
            u'appUserPastCnt',       u'appUserNewCnt',       
             u'clickOfDayCnt']
train=train[              'label',           'clickTime',      'conversionTime',            
                'creativeID',              'userID',          'positionID',      
            'connectionType',    'telecomsOperator',                'adID',      
                'campaignID',        'advertiserID',               'appID',      
               'appPlatform',         'appcategory',           'sitesetID',      
              'positionType',                 'age',              'gender',      
                 'education',      'marriageStatus',            'haveBaby',      
                  'hometown',           'residence',            'clickDay',      
            'clickTimeInDay',  'provinceOfHometown',      'provinceOfResidence', 'clickOfUserCnt',      
            'appUserPastCnt',       'appUserNewCnt',       
             'clickOfDayCnt']
train=train[              ['label',           'clickTime',      'conversionTime',            
                'creativeID',              'userID',          'positionID',      
            'connectionType',    'telecomsOperator',                'adID',      
                'campaignID',        'advertiserID',               'appID',      
               'appPlatform',         'appcategory',           'sitesetID',      
              'positionType',                 'age',              'gender',      
                 'education',      'marriageStatus',            'haveBaby',      
                  'hometown',           'residence',            'clickDay',      
            'clickTimeInDay',  'provinceOfHometown',      'provinceOfResidence', 'clickOfUserCnt',      
            'appUserPastCnt',       'appUserNewCnt',       
             'clickOfDayCnt']]
train.head()
train.to_csv('data/train_20.csv', index=False)
test = pd.read_csv('data/test_19.csv')
test.columns
test = test.drop(['Unnamed: 0'], axis=1)
test.columns
test=test.rename(columns={'appNewlyCnt':'appUserNewCnt'})
test=test.rename(columns={'appHistoryCnt':'appUserPastCnt'})
test=test.rename(columns={'province':'provinceOfHometown'})
test=test.rename(columns={'clickHistory':'clickOfUserCnt'})
test.columns
train.columns
test['provinceOfResidence']=np.floor(data['residence']/100)
test['provinceOfResidence']=np.floor(test['residence']/100)
test.head()
clickDayCntTest
clickDayCntTest=pd.DataFrame([[31,test.size]], columns={'clickDay','clickOfDayCnt'})
clickDayCntTest
clickDayCntTest=pd.DataFrame([[31,test.shape[0]]], columns={'clickDay','clickOfDayCnt'})
clickDayCntTest
clickDayCntTest=pd.DataFrame([[31,test.shape[0]]], columns={'clickOfDayCnt','clickDay'})
clickDayCntTest
ctest={'clickDay':[31], 'clickOfDayCnt':[test.shape[0]]}
ctest
clickDayCntTest=pd.DataFrame(ctest)
clickDayCntTest
test = test.join(clickDayCntTest.set_index('clickDay'), on = 'clickDay')
test.head()
test.columns
test=test['instanceID', 'label', 'clickTime', 'creativeID', 'userID',                    
       'positionID', 'connectionType', 'telecomsOperator', 'adID',               
       'campaignID', 'advertiserID', 'appID', 'appPlatform',                     
       'appcategory', 'sitesetID', 'positionType', 'age', 'gender',             
       'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence',      
       'clickDay', 'clickTimeInDay', 'provinceOfHometown',  'provinceOfResidence'                      
       'clickOfUserCnt', 'appUserPastCnt', 'appUserNewCnt',                       
       , 'clickOfDayCnt']    
test=test['instanceID', 'label', 'clickTime', 'creativeID', 'userID',                    
       'positionID', 'connectionType', 'telecomsOperator', 'adID',               
       'campaignID', 'advertiserID', 'appID', 'appPlatform',                     
       'appcategory', 'sitesetID', 'positionType', 'age', 'gender',             
       'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence',      
       'clickDay', 'clickTimeInDay', 'provinceOfHometown',  'provinceOfResidence' ,                     
       'clickOfUserCnt', 'appUserPastCnt', 'appUserNewCnt',                       
        'clickOfDayCnt']        
test=test[
['instanceID', 'label', 'clickTime', 'creativeID', 'userID',                    
       'positionID', 'connectionType', 'telecomsOperator', 'adID',               
       'campaignID', 'advertiserID', 'appID', 'appPlatform',                     
       'appcategory', 'sitesetID', 'positionType', 'age', 'gender',             
       'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence',      
       'clickDay', 'clickTimeInDay', 'provinceOfHometown',  'provinceOfResidence' ,                     
       'clickOfUserCnt', 'appUserPastCnt', 'appUserNewCnt',                       
        'clickOfDayCnt']        ]
test.columns
test.head()
test.to_csv('data/test_20.csv', index=False)
