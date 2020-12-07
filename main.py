
import numpy as np
from sklearn import neighbors

from util.file_util import FileUtil

if __name__ == "__main__":

    x = [[1,1], [2,2], [3,3], [4,4]]
    y = [1, 0, 0, 1]
    clf = neighbors.KNeighborsClassifier(2, weights='distance', metric="euclidean")
    clf.fit(x, y)

    print(clf.predict(np.array([[2, 3]])))




