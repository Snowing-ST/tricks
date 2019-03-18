def train_xgb(data_tfidf,tags,cv):
    '''
    调参步骤：
    1  学习率和树的棵树
    2. max_depth 和 min_weight
    3. gamma
    4. subsample 和 colsample_bytree 
    5. 正则化参数
    '''
    def gridcv(model,para):
        grid = GridSearchCV(model,param_grid = para,
                        scoring='accuracy',n_jobs=7,iid=False, cv=cv)
        grid.fit(data_tfidf, [i+1 for i in tags])
        print(grid.grid_scores_)
        print("The best parameters are %s with a score of %0.4f"
              % (grid.best_params_, grid.best_score_))    
        return grid
    
    
    estimator = XGBClassifier(
                # 模型参数
                n_estimators=400, 
                max_depth=6,min_child_weight=1,
                subsample=0.8,colsample_bytree=0.8,
                # 学习任务参数
                learning_rate =0.1,
                objective= 'multi:softmax',num_class=3, eval_metric = "merror",
                gamma=0,
    #            alpha=1,#L1正则化系数，默认为1
    #            lambda=1,#L2正则化系数，默认为1
                # 常规参数
                nthread=-1,scale_pos_weight=1, silent = 1,seed=27)   
    
    grid_values = {"n_estimators":range(100,500,100)}
    grid = gridcv(estimator,grid_values)
#    grid_values = {
#            'max_depth':[4,6,8],#用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。典型值：3-10
#            'min_child_weight':range(1,6,2),#用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。但是如果这个值过高，会导致欠拟合。
#                }
#    grid = gridcv(grid.best_estimator_,grid_values)
    grid_values = {
            'gamma':np.arange(0,3,0.5)#Gamma指定了节点分裂所需的最小损失函数下降值。参数的值越大，算法越保守。
                }
    grid = gridcv(grid.best_estimator_,grid_values)
#    grid_values = {
#             'subsample':[0.7,0.8,0.9],
#             'colsample_bytree':[0.7,0.8,0.9]
#                }
#    grid = gridcv(grid.best_estimator_,grid_values)
    grid_values = {
             'reg_alpha':[0.001,0.01,0.05,0.1, 1]
                }
    grid = gridcv(grid.best_estimator_,grid_values)
    return grid.best_estimator_
