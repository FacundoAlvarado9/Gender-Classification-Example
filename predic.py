from sklearn import tree, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#Dataset
#[Height, Weight, Shoe Size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

#Sex
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#Function
def PredictInput(toPredict):
    #Modelo: TREE
    clfTree = tree.DecisionTreeClassifier()
    clfTree = clfTree.fit(X,Y)

    print('Decision Tree says:', clfTree.predict(toPredict))

    #Modelo: SVM (Support Vectors)
    clfLSVM = svm.SVC(gamma='auto')
    clfLSVM.max_iter = 20000
    clfLSVM.fit(X, Y)

    print('SVM says:', clfLSVM.predict(toPredict))

    #Modelo: K-Neighbors
    clfKN = KNeighborsClassifier(n_neighbors=2)
    clfKN.fit(X,Y)

    print('KNeighbors says:', clfKN.predict(toPredict))

    #Modelo: Quadratic Analysis
    clfQA = QuadraticDiscriminantAnalysis()
    clfQA.fit(X,Y)

    print('Quadratic Discrimant says: ', clfQA.predict(toPredict))
    return

#Welcoming and asking for data
print('Welcome to the Gender Detector')
print('We work with some body measures, so we need you to answer some questions:')
print("What's your height?")
height = int(input())
print("What's your weight?")
weight = int(input())
print("What's your Shoe Size?")
shoeSize = int(input())

userInput = [[height, weight, shoeSize]]

#Predicting, by calling function
print(' ')
print('Predictions are:')
PredictInput(userInput)
