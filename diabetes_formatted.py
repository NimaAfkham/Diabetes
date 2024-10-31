import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree , DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from tkinter import *
import tkinter.ttk as ttk

def readData():
    data = pd.read_csv('diabetes.csv')
    return data

def splitData(data):
    x = data.values[:, 0:8]
    y = data.values[:, 8]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=100)
    return x, y, xTrain, xTest, yTrain, yTest

def train(xTrain , yTrain):
    forest = RandomForestClassifier(n_estimators = 1000, criterion='gini', random_state = 100)
    forest.fit(xTrain , yTrain)
    return forest

def prediction(xTest, forest):
    yPred = forest.predict(xTest)
    return yPred

def main_event( window ) :
    data = readData()

    df=pd.read_csv('diabetes.csv')

    df['Outcome'].value_counts()

    x=df.drop('Outcome',axis=1)
    y=df['Outcome']

    x=np.array(x)
    y=np.array(y)

    scaler=StandardScaler()
    x=scaler.fit_transform(x)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

    x_train.shape , y_train.shape

    x_test.shape , y_test.shape


    dt = DecisionTreeClassifier (max_depth=16 , min_samples_split=4 , min_samples_leaf=2)
    dt.fit(x_train,y_train)

    rf = RandomForestClassifier(n_estimators=200,max_depth=16, min_samples_split=4,min_samples_leaf=2)
    rf.fit(x_train,y_train)

    y_pred_train_dt = dt.predict(x_train)
    y_pred_train_rf = rf.predict(x_train)
    acc_train_dt=accuracy_score(y_train, y_pred_train_dt)
    acc_train_rf=accuracy_score(y_train, y_pred_train_rf)

    acc_train_dt,acc_train_rf

    y_pred_test_dt = dt.predict(x_test)
    y_pred_test_rf = rf.predict(x_test)
    acc_test_dt=accuracy_score(y_test, y_pred_test_dt)
    acc_test_rf=accuracy_score(y_test, y_pred_test_rf)

    acc_test_dt,acc_test_rf


    r_dt = recall_score(y_test, y_pred_test_dt)
    r_rf = recall_score(y_test, y_pred_test_rf)
    r_dt,r_rf

    p_dt = precision_score(y_test, y_pred_test_dt)
    p_rf = precision_score(y_test, y_pred_test_rf)
    p_dt,p_rf
    
    x, y, xTrain, xTest, yTrain, yTest = splitData(data)

    forest = train(xTrain, yTrain)
    pred = prediction([[Pregnancies.get(), Glucose.get() , BloodPressure.get() , Skinthickness.get() , Insulin.get() , Bmi.get() , DiabetesPedigreefunction.get() , Age.get() ]] , forest)
    result = StringVar()
    if pred == 1 :
        result = "Unfortunataley, You Have Diabetes"
    elif pred == 0 :
        result = "Luckily, You do not Have Diabetes"
    else :
        result = "Error"
    window.destroy()
    result_window = Tk()
    result_window.geometry( "%dx%d+%d+%d" % ( 750 , 100 , 300 , 50 ) )
    result_window.title("Result" )
    result_window.resizable(False,False)
    result_window.iconbitmap('icon.ico') 
    result_window.configure(background='white')
    Label(result_window, text = result , font = "Arial 25 bold" ).pack()

window = Tk()
window.geometry( "%dx%d+%d+%d" % ( 850 , 550 , 300 , 50 ) )
window.title("Diabetes Check" )
window.resizable(False,False)
window.iconbitmap('icon.ico') 
window.configure(background='white')
l1 = Label( window , text="Pregnancies" , justify = LEFT ,  font="tahoma 10 bold" , fg="black" , width=30  )
l1.place( x = 75 , y = 10  )
l2 = Label( window , text="Glucose" , justify = LEFT , font="tahoma 10 bold" , fg="black" , width=30  )
l2.place( x = 75 , y = 70 )
l3 = Label( window , text="BloodPressure" , justify = LEFT , font="tahoma 10 bold" , fg="black" , width=30  )
l3.place( x = 75 , y = 130 )
l4 = Label( window , text="Skinthickness" , justify = LEFT , font="tahoma 10 bold" , fg="black" , width=30  )
l4.place( x = 75 , y = 190  )
l5 = Label( window , text="Insulin" , justify = LEFT , font="tahoma 10 bold" , fg="black" , width=30 )
l5.place( x = 75 , y= 250  )
l6 = Label( window , text="BMI" , justify = LEFT ,  font="tahoma 10 bold" , fg="black" , width=30  )
l6.place( x = 75 , y = 310  )
l7 = Label( window , text="DiabetesPedigreefunction" , justify = LEFT , font="tahoma 10 bold" , fg="black" , width=30  )
l7.place( x = 75 , y = 370 )
l8 = Label( window , text="Age" , justify = LEFT , font="tahoma 10 bold" , fg="black" , width=30  )
l8.place( x = 75 , y = 430  )

Pregnancies = IntVar()
a1 = ttk.Combobox( window , textvariable=Pregnancies )
var1 = []
for i in range( 0,9) :
    var1.append(i)
a1["value"]=var1
a1.current(1)
a1.place( x = 450 , y = 10 )


Glucose = IntVar()
e1 = Scale( window , from_= 1 , to=900 , orient="horizontal" , activebackground="black" , cursor="dot" , length=400 , variable=Glucose)
e1.place( x=350 , y = 60 )

BloodPressure = IntVar()
e2 = Scale( window , from_= 1 , to=900 , orient="horizontal" , activebackground="black" , cursor="dot" , length=400 , variable=BloodPressure)
e2.place( x=350 , y = 115 )

Skinthickness = IntVar()
Skinthickness1 = Entry( window , textvariable=Skinthickness , font="tahoma 14 normal" , width=10 )
Skinthickness1.place( x = 450 , y = 190 )

Insulin = IntVar()
Insulin1 = Entry( window , textvariable=Insulin , font="tahoma 14 normal" , width=10 )
Insulin1.place( x=450 , y=250 )

Bmi = DoubleVar()
Bmi1 = Entry( window , textvariable=Bmi , font="tahoma 14 normal" , width=10 )
Bmi1.place( x = 450 , y = 310 )

DiabetesPedigreefunction = DoubleVar()
DiabetesPedigreefunction1 = Entry( window , textvariable=DiabetesPedigreefunction , font="tahoma 14 normal" , width=10 )
DiabetesPedigreefunction1.place( x=450 , y=370 )

Age = IntVar()
Age1 = Entry( window , textvariable=Age , font="tahoma 14 normal" , width=10 )
Age1.place( x=450 , y=430 )

Button( window , text="Confirm Data" , width=10  , command= lambda : main_event( window )).pack(side="bottom" , pady=20 )
       
window.mainloop()