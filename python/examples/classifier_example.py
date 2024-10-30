from fedtree import FLClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    X, y = load_digits(return_X_y=True)
    clf = FLClassifier(n_trees=10, mode="horizontal", n_parties=4, num_class=10, objective="multi:softmax")
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_prob = clf.predict_proba(X)
    accuracy = accuracy_score(y, y_pred)
    print("accuracy:", accuracy)
