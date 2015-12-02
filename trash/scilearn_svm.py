import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
classifier = svm.SVC(gamma=0.001,C  = 90)

l = len(digits.data)
learn = int(l * 0.7)
test = l - learn
x,y = digits.data[:learn], digits.target[:learn]
classifier.fit(x,y)

counter = 0
for i in xrange(1,test):
    predicted = classifier.predict(digits.data[-i])[0]
    real_value = digits.target[-i]
    if predicted != real_value:
        counter += 1
        print "predicted" ,predicted , "but real value is ", digits.target[-i]
        plt.imshow(digits.images[-i],cmap=plt.cm.gray_r,interpolation="nearest")
        plt.show()

print "Accuracy ", 1 - counter/(test+0.0)

