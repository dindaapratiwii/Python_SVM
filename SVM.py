#Ni Made Dinda Pratiwi
#1708561039
# Support Vector Machine
# Import libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Import datasets
from pandas import DataFrame

datasets = pd.read_csv('diabetes.csv')
X = datasets.iloc[:, [0,1,2,3,4,5,6,7]].values
Y = datasets.iloc[:, 8].values

# Splitting dataset kedalam Training set dan Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0) #Data training 75% dan Data tes 25%

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Memasukkan pengklasifikasi ke dalam set Pelatihan

from sklearn.svm import SVC #Import SVM model
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_Train, Y_Train) #Latih model menggunakan training set

# Memprediksi hasil set pengujian

Y_Pred = classifier.predict(X_Test)

# Membuat Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
c_matrix = confusion_matrix(Y_Test, Y_Pred)
ax = sns.heatmap(c_matrix, annot=True,
                 xticklabels=['No Diabetes','Diabetes'],
                 yticklabels=['No Diabetes','Diabetes'],
                 cbar=True, cmap='Blues', fmt='g')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()

# Visualising the Training set results

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Train, Y_Train
#membuat grid
X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
#garis yang menerapkan pengklasifikasi
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(6)]).T
pred = classifier.predict(Xpred).reshape(X1.shape)
plt.contourf(X1, X2, pred,
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
#memplot semua titik data
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ['red', 'green'][i], label = j)
plt.title('Support Vector Machine (Training set)')
plt.legend()
plt.show()

# Visualising the Test set results

from matplotlib.colors import ListedColormap
X_Set, Y_Set = X_Test, Y_Test

X1, X2 = np.meshgrid(np.arange(start = X_Set[:, 0].min() - 1, stop = X_Set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_Set[:, 1].min() - 1, stop = X_Set[:, 1].max() + 1, step = 0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(6)]).T
pred = classifier.predict(Xpred).reshape(X1.shape)
plt.contourf(X1, X2, pred,
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_Set)):
    plt.scatter(X_Set[Y_Set == j, 0], X_Set[Y_Set == j, 1],
                c = ['red', 'green'][i], label = j)
plt.title('Support Vector Machine (Test set)')
plt.legend()
plt.show()

# Model Accuracy
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(Y_Test, Y_Pred))

from sklearn.metrics import classification_report
print(classification_report(Y_Test, Y_Pred))