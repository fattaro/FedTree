from fedtree import FLClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file

if __name__ == '__main__':
    x, y = load_svmlight_file("../../dataset/a9a_modified")
    clf = FLClassifier(n_trees=50, mode="horizontal", n_parties=4, num_class=2)
    clf.fit(x, y)
    y_pred = clf.predict(x)
    y_pred_prob = clf.predict_proba(x)
    accuracy = accuracy_score(y, y_pred)
    print("accuracy:", accuracy)
