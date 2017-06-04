params = {
'min_child_weight': 1,
'eta': 0.01,
'colsample_bytree': 0.5,
'max_depth': 12,
'subsample': 0.8,
'alpha': 1,
'gamma': 1,
'silent': 1,
'verbose_eval': True,
'seed': RANDOM_STATE
}
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
xgtrain = xgb.DMatrix(X_train, label=y_train)
xgvalid = xgb.DMatrix(X_val, label=y_val)
xgtest = xgb.DMatrix(X_test)
#model = xgb.train(params, xgtrain, int(2012 / 0.9), feval=evalerror)
model = xgb.train(params, xgtrain, 200)
prediction=model.predict(xgtest)


from sklearn.metrics import roc_auc_score
eta = 0.2
max_depth = 5
subsample = 0.8
colsample_bytree = 0.8
start_time = time.time()

print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
params = {
"objective": "binary:logistic",
"booster" : "gbtree",
"eval_metric": "auc",
"eta": eta,
"tree_method": 'exact',
"max_depth": max_depth,
"subsample": subsample,
"colsample_bytree": colsample_bytree,
"silent": 1,
"seed": random_state,
 }
num_boost_round = 115
early_stopping_rounds = 10
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
print("Validating...")
check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)
score = roc_auc_score(X_valid[target].values, check)
 print('Check error value: {:.6f}'.format(score))

imp = get_importance(gbm, features)
 print('Importance array: ', imp)
 
 
 param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 6
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
num_round = 5
bst = xgb.train(param, xg_train, num_round, watchlist );
# get prediction
pred = bst.predict( xg_test );
print ('predicting, classification error=%f' % (sum( int(pred[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))
# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist );
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
yprob = bst.predict( xg_test ).reshape( test_Y.shape[0], 6 )
ylabel = np.argmax(yprob, axis=1)
print ('predicting, classification error=%f' % (sum( int(ylabel[i]) != test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))

