from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from the same folder as this script.
DATA_PATH = Path(__file__).resolve().parent / "kyphosis.csv"
df = pd.read_csv(DATA_PATH)
print(df.head())

# Plot pairwise relationships, grouped by the target class
sns.pairplot(df, hue='Kyphosis', palette='Set1')
plt.show()

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

pred = dtree.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# Render directly with scikit-learn; this does not require Graphviz.
tree.plot_tree(
	dtree,
	feature_names=list(X.columns),
	class_names=list(dtree.classes_),
	filled=True,
	rounded=True,
)
plt.tight_layout()
plt.show()

rfc = RandomForestClassifier(n_estimators=100, random_state=42)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))