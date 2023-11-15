# import dependencies.
import pandas as pd
import seaborn as sns
import pickle
import math
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    r2_score,
    mean_absolute_error as mae,
    mean_squared_error as mse,
)

# Storing dataset.
df = pd.read_csv('creditCard.csv')

# viewing basic details of dataset.
print("First 5 lines of Data frame \n",df.head())
print("Last 5 lines of Data frame \n",df.tail())
print("Number of Columns and rows in Data frame :",df.shape)
print(df.describe())
print("Number of missing values in Data frame :",df.isnull().sum().sum())
print(df.info())

# Number of frauds occured.
print("Total number of legit and fraudulant activities \n",df["Class"].value_counts())

# Total fruds and legit activities.
Class = df["Class"].value_counts()

# showing Number of legit and fraud activities in the dataset through bar plot
Class_df = pd.DataFrame({"status":Class.index,"values":Class.values})
sns.barplot(x = "status",y = "values",data = Class_df,width = 0.4)
plt.title("data set class division")
plt.show()

# storing number of legit and fraud transactions seperately.
legit = df[df.Class == 0]
fraud = df[df.Class == 1]
print("Shape of legit sample      :",legit.shape)
print("Shape of fraudulant sample :",fraud.shape)

# Statistical measures of the data.
# It gives logical idea behind the amount of transaction occured in 
# each fraudulant as well as legit action.
print("stats of legit amount\n",legit.Amount.describe())
print("Stats of fraudulant amount\n",fraud.Amount.describe())

# Correlation
cor_relation = df.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(cor_relation)
plt.show()

# undersampling the dataset. as number of legit sample are very high as compared to 
# fraudulant samples.
# Random taking legit samples of equal length of fraud samples.
legit_sample = legit.sample(n = 492,random_state = 0)
print("Legit sample after undersampling\n",legit_sample.head())
df2 = pd.concat([legit_sample,fraud],axis = 0)
print("head of legit sample\n",df2.head())
print("tail of legit sample\n",df2.tail())
Class = df2["Class"].value_counts()

# Number of legit and fraud sample after undersampling is shown in bar plot.
Class_df = pd.DataFrame({"status":Class.index,"values":Class.values})
sns.barplot(x = "status",y = "values",data = Class_df,width = 0.4)
plt.title("data set class division")
plt.show()

# defining dependent and independent variable.
# here class column is dependent variable and rest all are independent variable.
X = df2.drop(['Class'], axis = 1)
Y = df2["Class"]
xData = X.values
yData = Y.values
print(X.shape)
print(Y.shape)

# creating arrays to store different variations of all models.
# As we are taking 3 variations of each model. so we will store 
# each model in the memory 
lor_model = []
rfc_model = []
lr_model = []

# creating arrays to store accuracy score of different types of model.
# here classification report does'nt work linear regression model.so we will use 
# score() fuction to check accuracy score of linear regression model and both other 
# model will have a classification report.
lor_score = []
rfc_score = []
# we create an arrays in dictionary to store r2-score, mean squre
# error, root mean square error, and mean absolute error of linear
# regression model.
lr_score = {
     "r2_score":[],
     "mse":[],
     "rmse":[],
     "mae":[]
}


#Train and test data on different variation
# varition 1 : 80-20 split
# varition 2 : 75-25 split
# varition 3 : 70-30 split
for i in range(3):
    # spliting data into training and testing columns.
    xTrain, xTest, yTrain, yTest = train_test_split(xData, yData,
        test_size = 0.20 + (i*0.05), stratify = yData,random_state = 91)  

    # making use of Logistic regression model. 
    lor = LogisticRegression()                                      
    lor.fit(xTrain, yTrain)
    lor_pred = lor.predict(xTest)
    lor_model.append(lor)
    # Making classifcation report of logistic regression model.
    lor_score.append(classification_report(yTest, lor_pred))

    # making use of Random Forest Classifier  model. 
    rfc = RandomForestClassifier()                    
    rfc.fit(xTrain, yTrain)
    rfc_pred = rfc.predict(xTest)
    rfc_model.append(rfc)
    # Making classifcation report of Random Forest Classifier model.
    rfc_score.append(classification_report(yTest, rfc_pred))

    # making use of Linear regression model. 
    lr = LinearRegression()                                      
    lr.fit(xTrain, yTrain)
    lr_pred = lr.predict(xTest)
    lr_model.append(lr)
    # storing r2-score,mse,rmse and mae.
    lr_score["r2_score"].append(r2_score(yTest,lr_pred))
    lr_score["mse"].append(mse(yTest,lr_pred))
    lr_score["mae"].append(mae(yTest,lr_pred))
    # since, rmse is simply the square root of mse. so, we will just find square root
    # of mse using math function.
    lr_score["rmse"].append(math.sqrt(mse(yTest,lr_pred)))


#comparing all variatins of different models through bar plot
for i in range(3):
    list1=['linearRegression','LogisticRegression','RandomForest(entropy)']
    list2=[(lr_model[i]).score(xTest,yTest),(lor_model[i]).score(xTest,yTest),
           (rfc_model[i]).score(xTest,yTest)]
    df_Accuracy=pd.DataFrame({"Method Used":list1,"Accuracy":list2})
    chart=sns.barplot(x='Method Used',y='Accuracy',data=df_Accuracy,width = 0.6)
    # title vary depending on variation.
    plt.title(f"Split {80-(i*5)}-{20+(i*5)}")
    plt.show()

# printing accuracy score of different model in different variations.
for i in range(3):
    print(f"\nTrain Test split at {80-(i*5)}-{20+(i*5)}")
    print("Accuracy score of Linear regression model  :",(lr_model[i]).score(xTest,yTest))
    print("Accuracy score of Logistic regression model:",(lor_model[i]).score(xTest,yTest))
    print("Accuracy score of RandomForest model       :",(rfc_model[i]).score(xTest,yTest))
    
print("accuracy report of linear regression")

# printing Evaluation report of Linear regression (r2-score, mean_square_error,
# root_mean_sqare_error and mean_absolute_error.)
for i in range(3):
    print(f"\nTrain Test split at {80-(i*5)}-{20+(i*5)}")
    print("r2 score of Linear regression model              :",lr_score["r2_score"][i])
    print("mean square error of Linear regression model     :",lr_score["mse"][i])
    print("Root mean square error of Linear regression model:",lr_score["rmse"][i])
    print("mean absolute error of Linear regression model   :",lr_score["mae"][i])
               

# Making clasification report for different variations.(except linear regression).
for i in range(3):
    print(f"Train Test split at {80-(i*5)}-{20+(i*5)}")
    print("--Classification report of LogisticRegression model ",i+1,"--\n",(lor_score[i]))
for i in range(3):
    print(f"Train Test split at {80-(i*5)}-{20+(i*5)}")
    print("---Classification report of RandomForest model ",i+1,"---\n",(rfc_score[i]))

# dumping all the model to the pickle file.
# here we are Creating different pickle file for each model.
with open('lor_model.pkl', 'wb') as f:
    pickle.dump(lor_model, f)
with open('rfc_model.pkl', 'wb') as f:
    pickle.dump(rfc_model, f)
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

print(xTest[2])