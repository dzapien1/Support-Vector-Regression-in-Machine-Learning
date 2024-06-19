import matplotlib.pyplot as plt #print graphs
import csv #for works with csv files
from sklearn.svm import SVR #imports svr model
import numpy as np #scientific computer with python

#empy lists where is going to stock the data
dates = []
prices = []

#your csv path
path = r"C:\Use\Your\Path\Or\Directory"

#function that reads the csv and extracts the day of the month and adj price
def data(file):
    with open(file, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            #day of the month
            date = int(row[0].split('/')[1])
            dates.append(date)
            #adj price
            price = float(row[6])
            prices.append(price)
    return

#this function reshapes the dates ands create the svr models with the kernels
#trains the model and scatter plots the data, also creates the grapgh, labels, 
#display the plot and returns the prediction in the x value
def prices_prediction(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    
    svr_linear = SVR(kernel='linear', C=1e3)
    svr_polynomial = SVR(kernel='poly', C=1e3, degree=2)
    svr_radialbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_linear.fit(dates, prices)
    svr_polynomial.fit(dates, prices)
    svr_radialbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='DATA')
    plt.plot(dates, svr_radialbf.predict(dates), color='red', label='Radial Basis Function model')
    plt.plot(dates, svr_linear.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_polynomial.predict(dates), color='blue', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_radialbf.predict(np.array([[x]]))[0], svr_linear.predict(np.array([[x]]))[0], svr_polynomial.predict(np.array([[x]]))[0]

#calls the function and read csv file
data(path)

#predicts the prices using the 20 day
predicted_price = prices_prediction(dates, prices, 20)

#prints the predicted price
print(predicted_price)
