import numpy as np

w1 = np.zeros((20))
w2 = np.zeros((20))
#Входные вектора
D1 = np.genfromtxt('oddnumbers.csv', delimiter=',') #Нечетные числа 1 и 3
D2 = np.genfromtxt('even number.csv', delimiter=',') #Четные числа 0 и 2

Y1 = np.array([1,1]) # выходной нейрон для нечетных чисел
Y2 = np.array([0,0]) # для четных
 
alpha = 0.2
beta = - 0.4
σ = lambda x: 1 if x > 0 else 0

def f1(x, _w):
    s = beta + np.sum(x @ w1)
    return σ(s)
    print(s)
def f2(x, _w):
    s = beta + np.sum(x @ w2)
    return σ(s)
    print(s)
def train1(w1, D1, Y1):
    _w = w1.copy()
    for x, y in zip(D1, Y1):
        w1 += alpha * (y - f1(x, w1)) * x
    return(w1 != _w).any()
 
def train2(w2, D2, Y2):
    _w = w2.copy()
    for x, y in zip(D2, Y2):
        w2 += alpha * (y - f2(x, w2)) * x
    return (w2 != _w).any()

train1(w1, D1, Y1)
train2(w2, D2, Y2)
print(w1, w2)