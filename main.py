import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from plotkab import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    giniTree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    giniTree.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=giniTree, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.title('Gini, depth 4')
    plt.legend(loc='upper left')
    plt.show()

    giniTree2 = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=1)
    giniTree2.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=giniTree2, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.title('Gini, depth 6')
    plt.legend(loc='upper left')
    plt.show()

    entropyTree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
    entropyTree.fit(X_train, y_train)
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined, classifier=entropyTree, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.title('Entropy, depth 4')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.show()

    forest = RandomForestClassifier(criterion='gini', n_estimators=1, random_state=1, n_jobs=-1)
    forest.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.title('Random Forest, n 1')
    plt.ylabel('Szerokość płatka [cm]')
    plt.legend(loc='upper left')
    plt.show()

    forest2 = RandomForestClassifier(criterion='gini', n_estimators=5, random_state=1, n_jobs=-1)
    forest2.fit(X_train, y_train)
    plot_decision_regions(X_combined, y_combined, classifier=forest2, test_idx=range(105, 150))
    plt.xlabel('Długość płatka [cm]')
    plt.ylabel('Szerokość płatka [cm]')
    plt.title('Random Forest, n 5')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
