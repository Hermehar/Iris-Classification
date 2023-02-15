import numpy as np
import pandas as pd
import seaborn as sns
#creating dataset
from sklearn.datasets import load_iris
data = load_iris()
colNames = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']
df = pd.DataFrame(data=data.data, columns=colNames)
@np.vectorize
def getSpecies(x):
  return data.target_names[x]
df['species'] = getSpecies(data.target)
# analyzing data
# Dataset Summary
df.describe()
# Let us visualize the data
g = sns.pairplot(data=df, hue='species')
g.fig.suptitle("Graph 1", y=1.001)
# getting correlation
df.corr()
#training data model
X = df[colNames]
y = df.species
#using random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
model1 = RandomForestClassifier()
cross_val_score(model1, X, y).mean()
#using randomforest with different estimators
model2 = RandomForestClassifier(n_estimators=20)
cross_val_score(model2, X, y).mean()
#using SVM
model2 = RandomForestClassifier(n_estimators=20)
cross_val_score(model2, X, y).mean()
#using SVM with petal length and petal width only
model4 = SVC()
cross_val_score(model4, X[colNames[2:]], y).mean()
