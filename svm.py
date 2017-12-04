"""
ORIE 4741 - Final Project
SVM
Grace Mitchell
12/3/2017

Purpose of script is to complete 4 SVM models
"""

from sklearn import svm
import pandas as pd
import numpy as np

# import data

# courthouse
court_tr=pd.DataFrame.from_csv('courtnewf_train.csv')
court_tr=court_tr.values
court_te=pd.DataFrame.from_csv('courtnewf_test.csv')
court_te=court_te.values
court_va=pd.DataFrame.from_csv('courtnewf_valid.csv')
court_va=court_va.values
# office
office_tr=pd.DataFrame.from_csv('officenewf_train.csv')
office_tr=office_tr.values
office_te=pd.DataFrame.from_csv('officenewf_test.csv')
office_te=office_te.values
office_va=pd.DataFrame.from_csv('officenewf_valid.csv')
office_va=office_va.values
# school 1
school1_tr=pd.DataFrame.from_csv('school1newf_train.csv')
school1_tr=school1_tr.values
school1_te=pd.DataFrame.from_csv('school1newf_test.csv')
school1_te=school1_te.values
school1_va=pd.DataFrame.from_csv('school1newf_valid.csv')
school1_va=school1_va.values
# school 2
school2_tr=pd.DataFrame.from_csv('school2newf_train.csv')
school2_tr=school2_tr.values
school2_te=pd.DataFrame.from_csv('school2newf_test.csv')
school2_te=school2_te.values
school2_va=pd.DataFrame.from_csv('school2newf_valid.csv')
school2_va=school2_va.values

# courthouse
X_court=court_tr[:,:17]
y_court=court_tr[:,18]
court_valid=court_va[:,:17]
court_test=court_te[:,:17]
# Create SVM classification object 
court_model = svm.SVR(kernel='linear', C=1, gamma=1) 
court_model.fit(X_court, y_court)
court_model.score(X_court, y_court)
#Predict Output
court_valid_pred= court_model.predict(court_valid)
court_test_pred= court_model.predict(court_test)
# office
X_office=office_tr[:,:17]
y_office=office_tr[:,18]
office_valid=office_va[:,:17]
office_test=office_te[:,:17]
# Create SVM classification object 
office_model = svm.SVR(kernel='linear', C=1, gamma=1) 
office_model.fit(X_office, y_office)
office_model.score(X_office, y_office)
#Predict Output
office_valid_pred= office_model.predict(office_valid)
office_test_pred= office_model.predict(office_test)

# school 1
X_school1=school1_tr[:,:17]
y_school1=school1_tr[:,18]
school1_valid=school1_va[:,:17]
school1_test=school1_te[:,:17]
# Create SVM classification object 
school1_model = svm.SVR(kernel='linear', C=1, gamma=1) 
school1_model.fit(X_school1, y_school1)
school1_model.score(X_school1, y_school1)
#Predict Output
school1_valid_pred= school1_model.predict(school1_valid)
school1_test_pred= school1_model.predict(school1_test)

# school 2
X_school2=school2_tr[:,:17]
y_school2=school2_tr[:,18]
school2_valid=school2_va[:,:17]
school2_test=school2_te[:,:17]
# Create SVM classification object 
school2_model = svm.SVR(kernel='linear', C=1, gamma=1) 
school2_model.fit(X_school2, y_school2)
school2_model.score(X_school2, y_school2)
#Predict Output
school2_valid_pred= school2_model.predict(school2_valid)
school2_test_pred= school2_model.predict(school2_test)

# accuracy
# courthouse
court_valid_truth= court_va[:,18]
court_test_truth= court_te[:,18]
court_count_va=0
court_va_pdiff=np.zeros(len(court_valid_truth))
for i in range(len(court_valid_truth)):
    court_va_pdiff[i]=abs(court_valid_truth[i] - court_valid_pred[i])/court_valid_truth[i]
for i in range(len(court_valid_truth)):
    if court_va_pdiff[i] <= 0.1:
        court_count_va+=1
court_count_te=0
court_te_pdiff=np.zeros(len(court_test_truth))
for i in range(len(court_test_truth)):
    court_te_pdiff[i]=abs(court_test_truth[i] - court_test_pred[i])/court_test_truth[i]
for i in range(len(court_test_truth)):
    if court_te_pdiff[i]  <= 0.1:
        court_count_te+=1
court_valid_accuracy=court_count_va/len(court_valid_truth)
court_test_accuracy=court_count_te/len(court_test_truth)
print(court_valid_accuracy)
print(court_test_accuracy)

# office
office_valid_truth= office_va[:,18]
office_test_truth= office_te[:,18]
office_count_va=0
office_va_pdiff=np.zeros(len(office_valid_truth))
for i in range(len(office_valid_truth)):
    office_va_pdiff[i]=abs(office_valid_truth[i] - office_valid_pred[i])/office_valid_truth[i]
for i in range(len(office_valid_truth)):
    if office_va_pdiff[i] <= 0.1:
        office_count_va+=1
office_count_te=0
office_te_pdiff=np.zeros(len(office_test_truth))
for i in range(len(office_test_truth)):
    office_te_pdiff[i]=abs(office_test_truth[i] - office_test_pred[i])/office_test_truth[i]
for i in range(len(office_test_truth)):
    if office_te_pdiff[i]  <= 0.1:
        office_count_te+=1
office_valid_accuracy=office_count_va/len(office_valid_truth)
office_test_accuracy=office_count_te/len(office_test_truth)
print(office_valid_accuracy)
print(office_test_accuracy)

# school 1
school1_valid_truth= school1_va[:,18]
school1_test_truth= school1_te[:,18]
school1_count_va=0
school1_va_pdiff=np.zeros(len(school1_valid_truth))
for i in range(len(school1_valid_truth)):
    school1_va_pdiff[i]=abs(school1_valid_truth[i] - school1_valid_pred[i])/school1_valid_truth[i]
for i in range(len(school1_valid_truth)):
    if school1_va_pdiff[i] <= 0.1:
        school1_count_va+=1
school1_count_te=0
school1_te_pdiff=np.zeros(len(school1_test_truth))
for i in range(len(school1_test_truth)):
    school1_te_pdiff[i]=abs(school1_test_truth[i] - school1_test_pred[i])/school1_test_truth[i]
for i in range(len(school1_test_truth)):
    if school1_te_pdiff[i]  <= 0.1:
        school1_count_te+=1
school1_valid_accuracy=school1_count_va/len(school1_valid_truth)
school1_test_accuracy=school1_count_te/len(school1_test_truth)
print(school1_valid_accuracy)
print(school1_test_accuracy)

# school 2
school2_valid_truth= school2_va[:,18]
school2_test_truth= school2_te[:,18]
school2_count_va=0
school2_va_pdiff=np.zeros(len(school2_valid_truth))
for i in range(len(school2_valid_truth)):
    school2_va_pdiff[i]=abs(school2_valid_truth[i] - school2_valid_pred[i])/school2_valid_truth[i]
for i in range(len(school2_valid_truth)):
    if school2_va_pdiff[i] <= 0.1:
        school2_count_va+=1
school2_count_te=0
school2_te_pdiff=np.zeros(len(school2_test_truth))
for i in range(len(school2_test_truth)):
    school2_te_pdiff[i]=abs(school2_test_truth[i] - school2_test_pred[i])/school2_test_truth[i]
for i in range(len(school2_test_truth)):
    if school2_te_pdiff[i]  <= 0.1:
        school2_count_te+=1
school2_valid_accuracy=school2_count_va/len(school2_valid_truth)
school2_test_accuracy=school2_count_te/len(school2_test_truth)
print(school2_valid_accuracy)
print(school2_test_accuracy)

# concatenate
valid_a=[court_valid_accuracy,office_valid_accuracy,school1_valid_accuracy,school2_valid_accuracy]
print(np.mean(valid_a))
test_a=[court_test_accuracy,office_test_accuracy,school1_test_accuracy,school2_test_accuracy]
print(np.mean(test_a))