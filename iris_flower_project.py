#load libraries

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#load dataset

url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=read_csv(url,names=names)

#summarise the dataset

#shape (dimension)
print(dataset.shape)

#head (peek)
print(dataset.head(20))

#descriptions (statistical summary)
print(dataset.describe())

#class distribution (breakdown)
print(dataset.groupby('class').size())

#data visualisation

#box and whisker plots (univariate)
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()

#histogram (univariate)
dataset.hist()
plt.show()

#scatter plot matrix (multivariate)
scatter_matrix(dataset)
plt.show()

#split out validation dataset

array=dataset.values
X=array[:,0:4]
Y=array[:,4]
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=0.20,random_state=1)

#spot check algorithms

models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

#evaluate each model in turn

results=[]
names=[]

for name,model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
    cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s : %f (%f)'%(name,cv_results.mean(),cv_results.std()))

#we can see that SVM has largest estimated accuracy score i.e. 98%
#we can create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm bcos each algorithm was evaluated 10 times via 10 fold cross validation

#compare algorithms through the plots

plt.boxplot(results,labels=names)
plt.title('Algorithm Comparisons')
plt.show()

#make predictions on validation dataset

model = SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)

#evaluate predictions

print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

#accuracy around 96.67% on hold-out dataset





    








