###########################################################
###########################################################
####################                    ###################
####################   HP optimization  ###################
################# Suresh Kondati Natarajan ################
####################                    ###################
###########################################################
###########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler

#load n lines of data into pandas dataframe directly
n=29
df=pd.read_csv('set', sep=' ', header=1, skiprows=lambda i: i>n)
#print(df.head())
#print(df.describe())
print("Number of training points",len(df.index))
#[df.hist(i,bins=5) for i in df.columns]
#plt.show()
#[df.plot.scatter(i, 'RMSE', grid=True) for i in df.columns]
#plt.show()

# First polynomial basis must be created i.e. X1^2, X2^2, X1X2, etc..
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import StandardScaler

#considering polynomial order of 2 and 3 below
for orde in range(2,3):
 poly=PolynomialFeatures(orde,include_bias=False) 
 #poly=PolynomialFeatures(orde,include_bias=False,interaction_only=True) #interaction terms only for higher orders
 Xpoly = poly.fit_transform(df.drop('RMSE',axis=1))
 Xpoly_HP_name= poly.get_feature_names(df.columns)
 #option to scale data using StandardScaler
 #sc = StandardScaler()
 #Xpoly=sc.fit_transform(Xpoly)
 df_poly = pd.DataFrame(Xpoly, columns=Xpoly_HP_name)
 df_poly['RMSE']=df['RMSE']
 #print(df_poly.head())
 
 # no regularization 
 poly_model=LinearRegression(normalize=True)
 X_poly=df_poly.drop('RMSE', axis=1)
 y_poly=df_poly['RMSE']
 poly_model.fit(X_poly,y_poly)
 y_predict_poly=poly_model.predict(X_poly)
 RMSE_poly=np.sqrt(mean_squared_error(y_predict_poly,y_poly))
 print("RMSE of polynomial model of order",orde,":",RMSE_poly)
 coeff_poly = pd.DataFrame(poly_model.coef_,index=df_poly.drop('RMSE',axis=1).columns, columns=['Polynomial_Coefficients'])
 print(coeff_poly.count())
 coeff_poly.to_csv("poly"+str(orde)+"coefficients.fit")
 print ("R2 value of polynomial model of order",orde,":",poly_model.score(X_poly,y_poly))

 # include LassoCV regularization
 poly_model_R= LassoCV(cv=10,positive=False,verbose=1,normalize=True,eps=0.001,n_alphas=100,tol=0.0001,max_iter=500000,n_jobs=-1)
 poly_model_R.fit(X_poly,y_poly)
 y_predict_poly_R=poly_model_R.predict(X_poly)
 RMSE_poly_R=np.sqrt(mean_squared_error(y_predict_poly_R,y_poly))
 print("RMSE of polynomial metamodel",RMSE_poly_R)
 coeff_poly_R = pd.DataFrame(poly_model_R.coef_,index=df_poly.drop('RMSE',axis=1).columns,columns=['Metamodel_Coefficients'])
 coeff_poly_R_minimal=coeff_poly_R[coeff_poly_R['Metamodel_Coefficients']!=0]
 print(coeff_poly_R_minimal.count())
 coeff_poly_R.to_csv('meta'+str(orde)+'coeff.fit')
 coeff_poly_R_minimal.to_csv('meta'+str(orde)+'coeff-min.fit')
 print ("R2 value of polynomial metamodel of order",orde,":",poly_model_R.score(X_poly,y_poly))

 #plot y_linear vs y_predict_linear
 #plt.figure(figsize=(12,8))
 #plt.xlabel("Predicted value with linear fit",fontsize=20)
 #plt.ylabel("Actual y-values",fontsize=20)
 #plt.grid(1)
 #plt.scatter(y_predict_poly,y_poly,edgecolors=(0,0,0),lw=2,s=80)
 #plt.plot(y_predict_poly,y_predict_poly, 'k--', lw=2)
 #plt.show()

#for sets in range(1,11):
# #### Predict for new sets with names set1 set2 etc
# df2=pd.read_csv('set'+str(sets),sep=' ',header=1,skiprows=0)
# Xpoly2 = poly.fit_transform(df2.drop('RMSE',axis=1))
# df2_poly = pd.DataFrame(Xpoly2, columns=Xpoly_HP_name)
# df2_poly['RMSE']=df2['RMSE']
# X_poly2=df2_poly.drop('RMSE', axis=1)
# y_poly2=df2_poly['RMSE']
# y_predict_poly2=poly_model_R.predict(X_poly2)
# RMSE_poly2=np.sqrt(mean_squared_error(y_predict_poly2,y_poly2))
# print("RMSE of poly_R for set "+str(sets)+" :",RMSE_poly2)
# 
# #plot
# plt.figure(figsize=(12,8))
# plt.grid(1)
# plt.xlabel("Predicted RMSE",fontsize=20)
# plt.ylabel("Actual RMSE",fontsize=20)
# plt.title("Polynomial_R_order:"+str(orde)+" set "+str(sets))
# plt.ylim(0,1)
# plt.xlim(0,1)
# plt.scatter(y_predict_poly2,y_poly2,edgecolors=(0,0,0),lw=1,s=40)
# plt.plot(y_predict_poly2,y_predict_poly2, 'k--', lw=2)
# plt.show()

print('NN fits on original data')
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initializing Neural Network
NN_model = Sequential()
# Adding the input layer and the first hidden layer
NN_model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 9))
# Adding the second hidden layer
NN_model.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Adding the output layer
NN_model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
# Compiling Neural Network
NN_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#----------------scaling - uncomment if needed
sc=MinMaxScaler()
df[:]=sc.fit_transform(df[:])
print(df.head())
#------------------
X=df.drop('RMSE', axis=1)
y=df['RMSE']

# Fitting our model 
print('Fitting a NN model')
NN_model.fit(X, y, epochs = 1000)

y_predict_NN=NN_model.predict(X)
RMSE_NN=np.sqrt(mean_squared_error(y_predict_NN,y))
print("RMSE of NN model",RMSE_NN)


#plot
plt.figure(figsize=(12,8))
plt.grid(1)
plt.xlabel("Predicted RMSE",fontsize=20)
plt.ylabel("Actual RMSE",fontsize=20)
plt.title("NN-model")
plt.ylim(0,1)
plt.xlim(0,1)
plt.scatter(y_predict_NN,y,edgecolors=(0,0,0),lw=1,s=40)
plt.plot(y_predict_NN,y_predict_NN, 'k--', lw=2)
#plt.show()




fig=plt.figure(figsize=(8,12))

for sets in range(1,11):
 #### Predict for sets
 df2=pd.read_csv('set'+str(sets),sep=' ',header=1,skiprows=0)
 Xpoly2 = poly.fit_transform(df2.drop('RMSE',axis=1))
 df2_poly = pd.DataFrame(Xpoly2, columns=Xpoly_HP_name)
 df2_poly['RMSE']=df2['RMSE']
 X_poly2=df2_poly.drop('RMSE', axis=1)
 y_poly2=df2_poly['RMSE']
 df2[:]=sc.fit_transform(df2[:])
 X_NN=df2.drop('RMSE', axis=1)
 y_NN=df2['RMSE']
 NN_predict=NN_model.predict(X_NN)
 RMSE_poly2=np.sqrt(mean_squared_error(poly_model_R.predict(X_poly2),y_poly2))
 RMSE_poly2_WOR=np.sqrt(mean_squared_error(poly_model.predict(X_poly2),y_poly2))
 RMSE_NN=np.sqrt(mean_squared_error(NN_predict,y_NN))
 print("RMSE of poly   for set "+str(sets)+" :",RMSE_poly2_WOR)
 print("RMSE of poly_R for set "+str(sets)+" :",RMSE_poly2)
 print("RMSE of NN     for set "+str(sets)+" :",RMSE_NN)
 ax=fig.add_subplot(4,3,sets)
 #ax.hist(y_predict_poly2-y_poly2,bins=20,range=[-1,1], align='mid')
 #ax.hist(y_predict_poly2-y_poly2,bins=20, align='mid')
 NN_predict=np.transpose(NN_predict)[0]
 ax.hist(np.transpose(NN_predict)-y_NN,bins=20, align='mid')
 #ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(y_predict_poly2-y_poly2)))
 ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(NN_predict-y_NN)))
 ax1=ax.twinx()
 #ax1.scatter(y_predict_poly2,y_poly2,c='red',s=1)
 ax1.scatter(NN_predict,y_NN,c='red',s=1)
 #ax1.plot(y_poly2,y_poly2, 'k--', lw=2)
 ax1.plot(y_NN,y_NN, 'k--', lw=2)
 ax1.set_xlim(-1,1)
 #ax.set_ylim(0,1)
 #ax.set_title("order:"+str(orde)+" set: "+str(sets)+" RMSE: "+str(round(RMSE_poly2,2)))
 ax.set_title("order:"+str(orde)+" set: "+str(sets)+" RMSE: "+str(round(RMSE_NN,2)))
 #ax.set_xlabel("P")
 #ax.set_ylabel("A")

plt.tight_layout()
plt.show()


######################################
# PSO : pyswarm
######################################
#perform PSO optimization to find the minimum

from subprocess import getoutput
#import parser
out = getoutput("./make-eqn.sh")
print(out)
#aa=print(out.decode("utf=8"))
#code = parser.expr(aa).compile()

from pyswarm import pso
lb=[3,0.0,0.1,0.1,0.,0.,0,1,0.01]
ub=[6,1.0,0.7,0.7,0.16,0.16,2,10,0.1]

def func(x):
 X1=x[0]
 X2=x[1]
 X3=x[2]
 X4=x[3]
 X5=x[4]
 X6=x[5]
 X7=x[6]
 X8=x[7]
 X9=x[8]
 return eval(out)
# return  - 0.00727720*X1 + 0.05883694*X3 - 0.20047822*X4 + 0.05386346*X7 - 0.00179427*X1*X8 - 0.00010946*X2*X8 - 0.39085078*X3*X4 + 0.80357874*X3*X5 + 1.05332161*X3*X6 - 0.00007825*X3*X8 + 1.66539570*X3*X9 - 0.24099514*X4*X5 - 0.12930075*X4*X7 + 0.01263228*X4*X8 - 1.39071312*X4*X9 + 0.67109860*X5*X6 + 1.42485503*X5*X9 - 0.03994512*X6*X8 + 0.53353022*X7*X9 - 0.00616158*X8*X9

xopt, fopt = pso(func, lb, ub, swarmsize=100, maxiter=100, omega=0.5, phip=0.5, phig=0.5)

print(xopt)
