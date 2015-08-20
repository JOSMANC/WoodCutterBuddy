import pickle
import requests
import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class RemoveRealtorTestModel(object):
    def  __init__(self):
        self.features = None
        self.pca_model = None
        self.scorespriced = None
        self.scores = None
        self.features = None
        self.components = None
        self.log = None
        self.model = None        
        self.divided_position = None 

    def get_data(self, dataset='select * from '):
        '''
        '''        
        

    # def get_data(self, size=1., holdout=True):
    #     sql_query
    def init_final(self, features, components, log, divided_position, model):
        '''
        '''        
        self.features = features
        self.components = components
        self.log = log
        self.divided_position = divided_position
        self.model = model
        return None

    def fit(self, dfX, dfy):
        '''
        '''        
        start = time.time()
        dfX.reset_index(inplace=True,drop=True)
        boolvars = [col for col in dfX.columns.values if 'dvar' in col]
        if self.components > 0:
            self.pca_model = RandomizedPCA(n_components=self.components)
            self.pca_model.fit(dfX[boolvars].values)
            dftrain_bool = pd.DataFrame(self.pca_model.transform(dfX[boolvars].values))
            dftrain = pd.concat([dfX[self.features], dftrain_bool],axis=1)
        else:
            self.pca_model=None
            dftrain = dfX[self.features]
            
        if self.log:
            dftrain = self._log_feature(dftrain)
            dfy = np.log(dfy.values)
            self.log = True
        else:
            self.log = False

        if self.divided_position is not None:
            for i, j in self.divided_position:
                test_X = self._divide_two_features(test_X, i, j)
                train_X = self._divide_two_features(train_X, i, j)            
            
        self.model.fit(dftrain, dfy)
        print self.model.get_params
        print 'model fit {} homes in {}'.format(dfX.shape[0] ,time.time()-start)
        return None
    
    def predict(self, dfX, gettree):
        '''
        '''        
        start = time.time()
        dfX.reset_index(inplace=True,drop=True)
        boolvars = [col for col in dfX.columns.values if 'dvar' in col]

        if self.pca_model is not None:
            df_bool = pd.DataFrame(self.pca_model.transform(dfX[boolvars].values))            
            dftest = pd.concat([dfX[self.features], df_bool], axis=1)
        else:
            dftest = dfX[self.features]

        if self.log:
            dftest = self._log_feature(dftest)

        point_estimates = self.model.predict(dftest)
        tree_estimates = np.array([])
        
        if gettree:
            for est in self.model.estimators_:
                tree_estimates = np.concatenate([tree_estimates, est.predict(dftest)], axis=1)
        else:
            tree_estimates = 0
            
        print 'model predict (2x) {} in {}'.format(dfX.shape[0] ,time.time()-start)
        
        if self.log:
            return np.exp(point_estimates), np.exp(tree_estimates)
        else:
            return point_estimates, tree_estimates
            
    def df_time_iterator(self, df, cv, splittype):
        '''
        '''
        minlistdate = df.listdate.min()
        maxlistdate = df.statusupdate.max()
        dt = ((maxlistdate-minlistdate).days/cv)
        c = 1
        dayrandom = np.random.randint(0,16)-8
        while c < cv+1:
            #chunk does equal train/test splits
            if splittype == 'chunk':
                train = df[((df.listdate >= (minlistdate+timedelta(days=dt*(c-1)+dayrandom))) &
                           (df.statuschangedate <= (minlistdate+timedelta(days=dt*(c)))))].index

                test = df[((df.listdate > (minlistdate+timedelta(days=dt*(c)-dayrandom))) &
                          (df.statuschangedate <= (minlistdate+timedelta(days=dt*(c+1)))))].index
            
            #forward does train/test splits which grow in time
            elif splittype == 'forward':
                train = df[df.statuschangedate <= minlistdate+timedelta(days=dt*(c)-dayrandom)].index
                test = df[df.listdate > minlistdate+timedelta(days=dt*(c))].index
            c += 1
            yield train, test

    def cross_validate_model(self, dfX, dfy, features, components, log, divided_position, model, cv):
        '''
        '''        
        kf = KFold(dfX.shape[0], n_folds=cv, shuffle=True)
        dfX.reset_index(inplace=True, drop=True)
        dfy.reset_index(inplace=True, drop=True)
        mets = [metrics.median_absolute_error, metrics.r2_score, self.percent_difference]
        scores = np.zeros(len(mets))
        scorespriced = np.zeros(len(mets))
        boolvars = [col for col in dfX.columns.values if 'dvar' in col]
        if components>0:         
            features = features+boolvars 
        dfX = dfX[features]

        for train_index, test_index in kf:

            train_X, train_y = dfX.loc[train_index,:].copy(), dfy.loc[train_index].copy()
            test_X, test_y = dfX.loc[test_index,:].copy(), dfy.loc[test_index].copy()

            if components>0:
                train_X, test_X = self._pca_dummies(components, boolvars, train_X, test_X)

            if divided_position is not None:
                for i, j in divided_position:
                    test_X = self._divide_two_features(test_X, i, j)
                    train_X = self._divide_two_features(train_X, i, j)
                    
            self.features = train_X.columns.values
            
            if log:
                train_X = self._log_feature(train_X)
                train_y = np.log(train_y.values)
                test_X = self._log_feature(test_X)
                test_y = np.log(test_y.values)
            

            model.fit(train_X, train_y)

            ypred = model.predict(test_X)

            if log:
                test_y = np.exp(test_y)
                ypred = np.exp(ypred) 

            mask = (ypred>100000) & (ypred<230000)

            for i, m in enumerate(mets):
                scores[i] += m(test_y, ypred)
                scorespriced[i] +=  m(test_y[mask], ypred[mask])
        scores = scores/float(cv)
        scorespriced = scorespriced/float(cv)
        self.a_cved_model = model
        print ''.join(['-']*40)
        print ''.join(['-']*40)        
        print model.get_params
        print ''.join(['-']*40)       
        self._display_scoring_metrics(zip(mets,scores), 'full')
        print ''.join(['-']*40)
        self._display_scoring_metrics(zip(mets,scorespriced), 'priced')

        return None

    
    def scorer(self, ytrue, ypred):
        '''
        '''        
        mask = (ypred>100000) & (ypred<230000)
        mets = [metrics.median_absolute_error, metrics.r2_score, self._percent_difference]
        scores = np.zeros(len(mets))     
        scorespriced = np.zeros(len(mets))
        for i, m in enumerate(mets):
            scores[i] += m(ytrue, ypred)
            scorespriced[i] +=  m(ytrue[mask], ypred[mask])
        print 'full_size {}'.format(len(ytrue))
        self._display_scoring_metrics(zip(mets,scores), 'full')
        print ''.join(['-']*40)        
        print 'full_size {}'.format(np.sum(mask))        
        self._display_scoring_metrics(zip(mets,scorespriced), 'priced')           
        return None
        
    def percent_difference(self, ytest, ypred):
        '''
        '''        
        return np.mean(abs((ytest-ypred)/ytest)*100.)        

    
    def tree_importance(self, model, threshold):
        '''
        '''
        fimport = pd.DataFrame(zip(self.features, model.feature_importances_), columns=['feature','importance']).sort('importance', ascending=False)
        fvar=[]
        for est in model.estimators_:
            fvar.append(est.feature_importances_)
        fimport['std'] = np.array(fvar).std(axis=0)
        fimport['cumimport'] = fimport['importance'].cumsum().values
        return fimport, fimport[fimport['cumimport'] < threshold]['feature'].tolist()
    
    def _pca_dummies(self, components, boolvars, dftrain, dftest):
        '''
        '''
        dftrain.reset_index(inplace=True, drop=True)
        dftest.reset_index(inplace=True, drop=True)        
        self.pca_model = RandomizedPCA(n_components=components)
        self.pca_model.fit(dftrain[boolvars].values)

        dftrain_bool = pd.DataFrame(self.pca_model.transform(dftrain[boolvars].values))
        dftest_bool = pd.DataFrame(self.pca_model.transform(dftest[boolvars].values))
        
        dftrain.drop(boolvars, axis=1,inplace=True)
        dftest.drop(boolvars, axis=1,inplace=True)
        features = dftrain.columns.tolist()+['pca'+str(i) for i in range(components)]        
        dftrain = pd.concat([dftrain, dftrain_bool], axis=1, ignore_index=True)
        dftest = pd.concat([dftest, dftest_bool], axis=1, ignore_index=True)

        dftrain.columns = features
        dftest.columns = features  
        return dftrain, dftest

    def _display_scoring_metrics(self, met_scores, label):
        '''
        '''        
        for met, score in met_scores:
            print label+' {}: {}'.format(met.func_name, np.round(score,3))

    def _log_feature(self, df):
        '''
        '''
        for feature in [col for col in df.columns.values if (('price' in str(col)) |  ('taxes' in str(col)))]:
            df[feature] = np.log(df[feature].values)
        return df

    def _divide_two_features(self, df, f1, f2):
        '''
        '''
        df[f1+'_'+f2] = df[f1]/df[f2]
        df.drop([f1,f2], axis=1, inplace=True)
        return df

if __name__ == "__main__":
<<<<<<< HEAD
    features_to_try = []
    percent_train = 0.1
=======
    features_to_try = [
         'taxes',
         'livingarea',
         'time_params'         
         'approxlotsqft',
        'comp_median_closeprice_blockgroup180',
         'comp_meancloseprice_tract180',
         'comp_mediancloseprice_tract180',
         'comp_mediantaxes_tract180',
         'comp_meantaxes_tract180',
         'comp_median_taxes_blockgroup180',
         'comp_median_livingarea_blockgroup180',
         'comp_mean_taxes_blockgroup180',
         'comp_min_closeprice_blockgroup180',
         'comp_meanlivingarea_tract180',
         'comp_mincloseprice_tract180',
         'comp_mean_livingarea_blockgroup180',
         'comp_meanyearbuilt_tract180',
         'comp_mediandaytocontract_tract180',
         'comp_medianlivingarea_tract180',
         'comp_maxcloseprice_tract180']
    percent_train = 0.02
>>>>>>> 447cfe941b8219b1ed0c1f89fd036d06f3b59293
    model_list = [rfr1 = RandomForestRegressor(n_estimators=100, n_jobs=-1)
                  rfr2 = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
                  gbm1 = GradientBoostingRegressor(n_estimators= 5000, min_samples_split= 1,
                         learning_rate= 0.001, loss= 'ls')
                  gbm2 = GradientBoostingRegressor(n_estimators= 10000, min_samples_split= 1,
                         learning_rate= 0.001, loss= 'lad')]
    feature_list = []
    transformer_list = zip([],[])
    components = [1,2,4,5,6]
    log = [True, False]
    rrtm = RemoveRealtorTestModel()
    dftest, dftest_y,  dfhold, dfhold_y = rrtm.get_data()
    smaller_index = (np.random.rand(dftest.shape[0]) < percent_train)

    print 'rows ', smaller_index.sum()
    for mod in [model_list]:
        for mod in []
        starttime = time.time()
        rrtm.cross_validate_model(dftest[smaller_index].copy(), 
                              dftest_y[smaller_index].copy(), 
                              features=, components=, log=, model=, cv=)
        print time.time()-starttime, ' time'

    rrtm.fit( dftest.copy(), dftest_y.copy(), features=, components=, log=, model=)
    print rrtm.predict(dftest.head(2).copy())
