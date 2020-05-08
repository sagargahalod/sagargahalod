from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

dataset=datasets.load_iris()
model= GaussianNB()

model.fit(dataset.data,dataset.target)
exempt=dataset.target

predicted =model.predict(dataset.data)
print(metrics.classification_report(exempt,predicted))
print(metrics.confusion_matrix(exempt,predicted))