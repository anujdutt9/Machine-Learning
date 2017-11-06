import numpy as np
from collections import Counter,defaultdict

# Chills: Y=1, N=0
# RunnyNose: Y=1, N=0
# Headache: No=0, Mild=1, Strong=2
# Fever: Y=1, N=0
#  Flu: Y=1, N=0
X = np.asarray(((1, 0, 1, 1),
                (1, 1, 0, 0),
                (1, 0, 2, 1),
                (0, 1, 1, 1),
                (0, 0, 0, 0),
                (0, 1, 2, 1),
                (0, 1, 2, 0),
                (1, 1, 1, 1)))

y = np.asarray((0, 1, 1, 1, 0, 1, 0, 1))

# Class Count
y_count = []
for val in Counter(y).values():
    y_count.append(val)
print('y_count: ', y_count)




# Prior Probability P(H)
def prior_probb(classes):
    tot = len(classes)
    prob = dict(Counter(classes))
    print('prob: ',prob)
    for key in prob.keys():
        prob[key] = prob[key]/tot
    return prob



# P(Chills=Y|Flu=Y)
def p_X_given_y(X,col,y_n):
    x = X[:, col]
    tot1 = 0
    for i in range(len(x)):
        if (x[i] == y_n) and (y[i] == 1):
            tot1 += 1
    return tot1



# P(E|H)
def likelihood(X,i,yes_no):
    tot1 = 0
    tot1 = p_X_given_y(X,col=i,y_n=yes_no)
    return (tot1 / float(y_count[1]))



P_c = prior_probb(y)[1]
print('P(c): ',P_c)

a = []
yesNo = [1,0,1,0]
for i in range(X.shape[1]):
    p_yes = likelihood(X,i,yes_no=yesNo[i])
    a.append(p_yes)
    print('P(H=Yes/No|Flu=Yes) = ', p_yes)

print('a: ',a)

b = []
for i in range(X.shape[1]):
    x = X[:, i]
    probb = prior_probb(x)
    b.append(probb)
print('b: ',b)

b_prod = []
P_chills = b[0][1]
b_prod.append(P_chills)
print('P chills: ',P_chills)
P_runnyNose = b[1][0]
b_prod.append(P_runnyNose)
print('P runnyNose: ',P_runnyNose)
P_headache = b[2][1]
b_prod.append(P_headache)
print('P headache: ',P_headache)
P_fever = b[3][0]
b_prod.append(P_fever)
print('P fever: ',P_fever)


print('product: ',np.product(a))
print('product b: ',np.product(b_prod))

u = float(P_c * np.product(a))/(float(np.product(b_prod)))
print('Naive Bayes with Denominator [Flu=Yes]: ',u)

print('Naive Bayes without Denominator [Flu=Yes]: ',float(P_c * np.product(a)))
