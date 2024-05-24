
# %% import library
import math
import numpy as np
import matplotlib.pyplot as plt

# %%


def fx(a, b, h=500):
    X = []
    Y = []
    for i in range(0,h+1):
        x = a + (1.0*(b-a)/h)*i
        y =  math.sin(x)

        
        X.append(x)
        Y.append(y)

    return X, Y


# %%
# start from zero
# n = 5

pointCount = 4
point = pointCount+1

x = np.array([1, 2, 2.8, 3, 4, 5])
y = np.array([2, 4, 2, 5, 6, 8])

x, y = fx(0, 2.0 * math.pi, pointCount)
plt.grid()
plt.plot(x, y, 'o')
# %%


def aj(j):
    return y[j]


def hi(i):
    return x[i+1] - x[i]


def sj(X, j, bj, cj, dj):
    ajj = aj(j)

    diff = (X - x[j])

    return ajj + bj[j] * diff + cj[j] * (diff**2) + dj[j] * (diff ** 3)


# %%
# create A matrix
A = []
a = [0] * (point)
a[0] = 1

A.append(a)


currentRow = 1
index = 0
for i in range(point-2):
    rowI = [0] * (point)

    rowI[index] = hi(index)
    rowI[index + 1] = 2 * (hi(index) + hi(index + 1))
    rowI[index + 2] = hi(index + 1)

    A.append(rowI)

    index += 1
    currentRow += 1

print(index)

lastRow = [0] * (point)

lastRow[-1] = 1


A.append(lastRow)
print(A)

# %%
# create matrix B

B = []
B.append(0)

unknownIndex = 1
for i in range(1, point-1):

    unknown1 = (3.0/hi(i)) * (aj(i + 1) - aj(i))
    unknown2 = (-3.0/hi(i-1)) * (aj(i) - aj(i-1))

    B.append(unknown1 + unknown2)

    unknownIndex += 1

B.append(0)

print(B)
# %%

Anumpy = np.array(A)
BNumpy = np.array(B)

print(Anumpy)
print(BNumpy)

# %%

ci = np.linalg.solve(Anumpy, BNumpy)

print(ci)
# %%

bj = []

for i in range(point-1):
    firstTerm = (1/hi(i)) * (aj(i+1) - aj(i))
    secondTerm = (-hi(i)/3.0) * (2 * ci[i] + ci[i+1])
    bj.append(firstTerm + secondTerm)

print(bj)
# %%
dj = []

for i in range(point-1):
    firstTerm = (1/(3*hi(i))) * (ci[i+1] - ci[i])

    dj.append(firstTerm)

print(dj)
# %%
XApprox = []
YApprox = []

h = 500
for i in range(point-1):
    xi = x[i]
    xNext = x[i+1]

    dis = (xNext - xi)/h
    for j in range(h):
        newX = xi + j * dis
        newY = sj(newX, i, bj, ci, dj)
        XApprox.append(newX)
        YApprox.append(newY)


# %%
X,Y = fx(0,2.0 * math.pi,500)
plt.plot(X,Y,color = 'black')
plt.plot(x, y, 'o')
xarr = np.array(XApprox)
yarr = np.array(YApprox)
plt.plot(xarr, yarr, color='red')
plt.grid()
plt.show()
# %%
