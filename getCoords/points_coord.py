# IMPORTS

import numpy as np
from collections import Counter


# FUNCTIONS


def calcul_coord(X):

    # CONST

    result = []

    # Get width and diag
    temp = []
    for i in range(0, len(X)-1):
        for y in range(i+1, len(X)):
            len1 = np.sqrt(pow((X[y][0] - X[i][0]), 2) + pow((X[y][1] - X[i][1]), 2))
            temp.append(len1)

    occur = Counter(temp)

    width = min(occur)
    occur.pop(width)
    diag = min(occur)

    # Get base point
    baseP = X[0]

    for i in X:
        if i[0] < baseP[0]:
            baseP = i
        elif i[0] == baseP[0] and i[1] < baseP[1]:
            baseP = i

    # Generate result
    nextLine = False

    while True:
        neigh = []
        for y in range(0, len(X)):
            if np.sqrt(pow((X[y][0] - baseP[0]), 2) + pow((X[y][1] - baseP[1]), 2)) == 0:
                p0 = X[y]
            if np.sqrt(pow((X[y][0] - baseP[0]), 2) + pow((X[y][1] - baseP[1]), 2)) == width:
                neigh.append(X[y])
            if np.sqrt(pow((X[y][0] - baseP[0]), 2) + pow((X[y][1] - baseP[1]), 2)) == diag:
                p1 = X[y]
        X.remove(p0)

        if len(neigh) == 2:
            if neigh[0][1] < neigh[1][1]:
                if not nextLine:
                    nextLine = neigh[1]
                baseP = neigh[0]
            else:
                if not nextLine:
                    nextLine = neigh[0]
                baseP = neigh[1]
            result.append({'corner1': p0, 'corner2': p1, 'arenaCenter': [(p0[0]+p1[0])/2, (p0[1]+p1[1])/2]})
        else:
            if nextLine:
                baseP = nextLine
                nextLine = False
            else:
                break

    return result

# TEST


'''
c1 = [0, 0]
c2 = [0, 1]
c3 = [1, 0]
c4 = [1, 1]
c5 = [2, 0]
c6 = [2, 1]
c7 = [3, 0]
c8 = [3, 1]
c9 = [0, 2]
c10 = [1, 2]
c11 = [2, 2]
c12 = [3, 2]
'''

'''
c1 = [4, 3]
c2 = [2, 3]
c3 = [3, 4]
c4 = [1, 2]
c5 = [3, 2]
c6 = [2, 1]
'''

c1 = [2, 4]
c2 = [1, 3]
c3 = [3, 3]
c4 = [2, 2]
c5 = [3, 1]
c6 = [4, 2]

X = [c2, c3, c1, c4, c6, c5]

print(calcul_coord(X))
