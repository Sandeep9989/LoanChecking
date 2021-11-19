import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import pickle as pck
from sklearn.metrics import confusion_matrix


df= pd.read_csv('LoanData.csv')
print(df.columns)

print(df['Gender'].unique())
print(df['Married'].unique())
print(df['Dependents'].unique())
print(df['Education'].unique())
print(df['Self_Employed'].unique())
print(df['ApplicantIncome'].unique())
print(df['CoapplicantIncome'].unique())
print(df['LoanAmount'].unique())
print(df['Loan_Amount_Term'].unique())
print(df['Credit_History'].unique())
print(df['Property_Area'].unique())
print(df['Loan_Status'].unique())

# df=df.drop(["Loan_ID"],axis=1)
# print(df.columns)

#changing the character to numerical type , Y =1 and N=0
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N":0})
#Self_Employed column have the Yes and No values , there are same value with different case sensitive
#replacing the yes with "Yes" and no with "No"
df["Self_Employed"] = df["Self_Employed"].replace(['yes'], 'Yes')
df["Self_Employed"] = df["Self_Employed"].replace(['no'], 'No')

df["Education"] = df["Education"].replace(['Not Graduate'], 'Not_Graduate')

triaing_df = df.drop(["Loan_ID","Loan_Status"],axis=1)
# print("--------------------------")
# print(triaing_df)
#
testing_df = df["Loan_Status"]
# print("--------------------------")
# print(testing_df)


train_df_cht = pd.get_dummies(triaing_df,drop_first=True)
# print(train_df_encoded.head())
print(train_df_cht.columns)
print(train_df_cht.head())

train_df_cht = train_df_cht.rename(columns={"Dependents_3+":"Dependents_moreth3"})
print(train_df_cht.columns)

X_train,X_test,Y_train,Y_test=train_test_split(train_df_cht,testing_df,random_state=42,test_size=0.3)


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, Y_train)
X_test_pred = sgd_clf.predict(X_test)
print(X_test_pred)

cm = confusion_matrix(Y_test, X_test_pred)
print(cm)


pck.dump(sgd_clf,open("LoanApproval.sav","wb"))

Xtest_decision_values = sgd_clf.decision_function(X_test)
print(Xtest_decision_values)

from sklearn.metrics import roc_auc_score as auc_score
auc_value = auc_score(Y_test, Xtest_decision_values)
print(auc_value)