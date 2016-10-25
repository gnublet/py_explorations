#AUTHOR: Kevin

## CHALLENGE - create 3 more classifiers...
#1 Random Forest
#2 Multi-layer Perceptron
#3 Support Vector Machine

#comparison 
#Try cross validation since we have a small dataset
from sklearn.model_selection import cross_val_score
#enable preprocessing (for MLP) under cross-validation
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

from sklearn import tree#decision tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

#Data: [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#mytest = [165,54,40]#me
mytest = [190,70,43]
#CHALLENGE - ...and train them on our data

#Decision Tree
nfolds = 5
myscores = {}

clf_DT = tree.DecisionTreeClassifier()
#do k=5 fold crossvalidation
scores_DT = cross_val_score(clf_DT, X,Y, cv = nfolds)#cv = number of folds (how to choose?)
# I was thinking i want to maximize the number of folds since our dataset is so small
print("Decision Tree scores: ",scores_DT)
print("Accuracy: %0.2f (+/- %0.2f)" %(scores_DT.mean(), scores_DT.std()*2))
myscores['DT'] = scores_DT.mean()

#Random Forest
#using the same training data
clf_RF = RandomForestClassifier(n_estimators = 10)#how to choose n_estimators?
#cv
scores_RF = cross_val_score(clf_RF, X,Y, cv = nfolds)
print("Random Forest scores: ",scores_RF)
print("Accuracy: %0.2f (+/- %0.2f)" %(scores_RF.mean(), scores_RF.std()*2))
myscores['RF'] = scores_RF.mean()

#Multi-layer perceptron
#MLP is sensitive to feature scaling on the data so lets standardize 
clf_MLP = make_pipeline(preprocessing.StandardScaler(), MLPClassifier(solver= 'lbfgs', alpha = 1e-5, hidden_layer_sizes=(5,2), random_state=1))
scores_MLP = cross_val_score(clf_MLP, X, Y, cv = nfolds)
print("Multilayer Perceptron scores: ",scores_MLP)
print("Accuracy: %0.2f (+/- %0.2f)" %(scores_MLP.mean(), scores_MLP.std()*2))
myscores['MLP'] = scores_MLP.mean()

#Support Vector Machine
clf_SVM = svm.SVC()
scores_SVM = cross_val_score(clf_SVM, X,Y,cv = nfolds)
print("SVM scores: ",scores_SVM)
print("Accuracy: %0.2f (+/- %0.2f)" %(scores_SVM.mean(), scores_SVM.std()*2))
myscores['SVM'] = scores_SVM.mean()
########

print('')
#Special Case: ME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (I'm a skinny guy)

#Decision Tree predictions
clf_DT = clf_DT.fit(X, Y)

prediction_DT = clf_DT.predict([mytest])
#prediction = clf_DT.predict([[165, 54, 40]])
#CHALLENGE compare their results and print the best one!
print("The decision tree classifier predicts {} to be {}.".format(mytest, prediction_DT))

#Random Forest predictions
clf_RF = clf_RF.fit(X,Y)
prediction_RF = clf_RF.predict([mytest])
print("The random forest classifier predicts {} to be {}.".format(mytest, prediction_RF))

#MLP predictions
#clf_MLP = clf_MLP.fit(X,Y)
#prediction_MLP = clf_MLP.predict([mytest])
#print("The multi-layer perceptron classifier predicts {} to be {}.".format(mytest, prediction_MLP))
y_test = ['male']
scaler = preprocessing.StandardScaler().fit(X)#Scale X
X_transformed = scaler.transform(X)
clf_MLP = clf_MLP.fit(X_transformed, Y)
X_test_transformed = scaler.transform([mytest])
#print(clf_MLP.score(X_test_transformed, y_test))
prediction_MLP = clf_MLP.predict(X_test_transformed)
print("The multilayer perceptron classifier predicts {} to be {}.".format(mytest, prediction_MLP))

#SVM predictions
clf_SVM = clf_SVM.fit(X,Y)
prediction_SVM = clf_SVM.predict([mytest])
print("The support vector classifier predicts {} to be {}.".format(mytest, prediction_SVM))

#print("SVM support vectors:\n", clf_SVM.support_vectors_)
#print("SVM support indices:", clf_SVM.support_)
#print("number of support vectors for each class:", clf_SVM.n_support_)
#print(mytest, X_test_transformed)

#which is most accurate?
print('')

mymax = 0
mymodel = ''
for k in myscores:
	if myscores[k]>=mymax:
		mymax = myscores[k]
		mymodel = k
	#print(k, myscores[k])

print("The most accurate model is {} with an accuracy of {}".format(mymodel, mymax))
