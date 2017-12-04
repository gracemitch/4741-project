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
import matplotlib.pyplot as plt

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
court_va_diff=np.zeros(len(court_valid_truth))
for i in range(len(court_valid_truth)):
    court_va_pdiff[i]=abs(court_valid_truth[i] - court_valid_pred[i])/court_valid_truth[i]
    court_va_diff[i]=court_valid_truth[i] - court_valid_pred[i]
for i in range(len(court_valid_truth)):
    if court_va_pdiff[i] <= 0.1:
        court_count_va+=1
court_count_te=0
court_te_pdiff=np.zeros(len(court_test_truth))
court_te_diff=np.zeros(len(court_test_truth))
for i in range(len(court_test_truth)):
    court_te_pdiff[i]=abs(court_test_truth[i] - court_test_pred[i])/court_test_truth[i]
    court_te_diff[i]=court_test_truth[i] - court_test_pred[i]
for i in range(len(court_test_truth)):
    if court_te_pdiff[i]  <= 0.1:
        court_count_te+=1
court_valid_accuracy=court_count_va/len(court_valid_truth)
court_test_accuracy=court_count_te/len(court_test_truth)
# MSE/RMSE
court_va_mse=np.mean(court_va_diff**2)
court_te_mse=np.mean(court_te_diff**2)
court_va_rmse=np.sqrt(court_va_mse)
court_te_rmse=np.sqrt(court_te_mse)
print(court_valid_accuracy)
print(court_test_accuracy)

# office
office_valid_truth= office_va[:,18]
office_test_truth= office_te[:,18]
office_count_va=0
office_va_pdiff=np.zeros(len(office_valid_truth))
office_va_diff=np.zeros(len(office_valid_truth))
for i in range(len(office_valid_truth)):
    office_va_pdiff[i]=abs(office_valid_truth[i] - office_valid_pred[i])/office_valid_truth[i]
    office_va_diff[i]=office_valid_truth[i] - office_valid_pred[i]
for i in range(len(office_valid_truth)):
    if office_va_pdiff[i] <= 0.1:
        office_count_va+=1
office_count_te=0
office_te_pdiff=np.zeros(len(office_test_truth))
office_te_diff=np.zeros(len(office_test_truth))
for i in range(len(office_test_truth)):
    office_te_pdiff[i]=abs(office_test_truth[i] - office_test_pred[i])/office_test_truth[i]
    office_te_diff[i]=office_test_truth[i] - office_test_pred[i]
for i in range(len(office_test_truth)):
    if office_te_pdiff[i]  <= 0.1:
        office_count_te+=1
office_valid_accuracy=office_count_va/len(office_valid_truth)
office_test_accuracy=office_count_te/len(office_test_truth)
# MSE/RMSE
office_va_mse=np.mean(office_va_diff**2)
office_te_mse=np.mean(office_te_diff**2)
office_va_rmse=np.sqrt(office_va_mse)
office_te_rmse=np.sqrt(office_te_mse)
print(office_valid_accuracy)
print(office_test_accuracy)

# school 1
school1_valid_truth= school1_va[:,18]
school1_test_truth= school1_te[:,18]
school1_count_va=0
school1_va_pdiff=np.zeros(len(school1_valid_truth))
school1_va_diff=np.zeros(len(school1_valid_truth))
for i in range(len(school1_valid_truth)):
    school1_va_pdiff[i]=abs(school1_valid_truth[i] - school1_valid_pred[i])/school1_valid_truth[i]
    school1_va_diff[i]=school1_valid_truth[i] - school1_valid_pred[i]
for i in range(len(school1_valid_truth)):
    if school1_va_pdiff[i] <= 0.1:
        school1_count_va+=1
school1_count_te=0
school1_te_pdiff=np.zeros(len(school1_test_truth))
school1_te_diff=np.zeros(len(school1_test_truth))
for i in range(len(school1_test_truth)):
    school1_te_pdiff[i]=abs(school1_test_truth[i] - school1_test_pred[i])/school1_test_truth[i]
    school1_te_diff[i]=school1_test_truth[i] - school1_test_pred[i]
for i in range(len(school1_test_truth)):
    if school1_te_pdiff[i]  <= 0.1:
        school1_count_te+=1
school1_valid_accuracy=school1_count_va/len(school1_valid_truth)
school1_test_accuracy=school1_count_te/len(school1_test_truth)
# MSE/RMSE
school1_va_mse=np.mean(school1_va_diff**2)
school1_te_mse=np.mean(school1_te_diff**2)
school1_va_rmse=np.sqrt(school1_va_mse)
school1_te_rmse=np.sqrt(school1_te_mse)
print(school1_valid_accuracy)
print(school1_test_accuracy)

# school 2
school2_valid_truth= school2_va[:,18]
school2_test_truth= school2_te[:,18]
school2_count_va=0
school2_va_pdiff=np.zeros(len(school2_valid_truth))
school2_va_diff=np.zeros(len(school2_valid_truth))
for i in range(len(school2_valid_truth)):
    school2_va_pdiff[i]=abs(school2_valid_truth[i] - school2_valid_pred[i])/school2_valid_truth[i]
    school2_va_diff[i]=school2_valid_truth[i] - school2_valid_pred[i]
for i in range(len(school2_valid_truth)):
    if school2_va_pdiff[i] <= 0.1:
        school2_count_va+=1
school2_count_te=0
school2_te_pdiff=np.zeros(len(school2_test_truth))
school2_te_diff=np.zeros(len(school2_test_truth))
for i in range(len(school2_test_truth)):
    school2_te_pdiff[i]=abs(school2_test_truth[i] - school2_test_pred[i])/school2_test_truth[i]
    school2_te_diff[i]=school2_test_truth[i] - school2_test_pred[i]
for i in range(len(school2_test_truth)):
    if school2_te_pdiff[i]  <= 0.1:
        school2_count_te+=1
school2_valid_accuracy=school2_count_va/len(school2_valid_truth)
school2_test_accuracy=school2_count_te/len(school2_test_truth)
# MSE/RMSE
school2_va_mse=np.mean(school2_va_diff**2)
school2_te_mse=np.mean(school2_te_diff**2)
school2_va_rmse=np.sqrt(school2_va_mse)
school2_te_rmse=np.sqrt(school2_te_mse)
print(school2_valid_accuracy)
print(school2_test_accuracy)

# concatenate
valid_a=[court_valid_accuracy,office_valid_accuracy,school1_valid_accuracy,school2_valid_accuracy]
test_a=[court_test_accuracy,office_test_accuracy,school1_test_accuracy,school2_test_accuracy]
valid_mse=np.mean([court_va_mse,office_va_mse,school1_va_mse,school2_va_mse])
test_mse=np.mean([court_te_mse,office_te_mse,school1_te_mse,school2_te_mse])
valid_rmse=np.mean([court_va_rmse,office_va_rmse,school1_va_rmse,school2_va_rmse])
test_rmse=np.mean([court_te_rmse,office_te_rmse,school1_te_rmse,school2_te_rmse])
print(valid_mse)
print(test_mse)
print(valid_rmse)
print(test_rmse)

# plots
# can change kernel, C, gamma (poly, rbf, sigmoid)
# kernel = linear, poly, sigmoid
# C range = 10^-5, 10, 10^5

# linear kernel function
C_range=[0.01,1,100]
court_valid_accuracy_lin=[0.1148105625717566*100,0.799081515499426*100,0.8019517795637199*100]
office_valid_accuracy_lin=[0.33696900114810563*100,0.9202066590126292*100,0.9225028702640643*100]
school1_valid_accuracy_lin=[0.2766934557979334*100,0.8438576349024111*100,0.8438576349024111*100]
school2_valid_accuracy_lin=[0.1452353616532721*100,0.7910447761194029*100,0.7893226176808267*100]

court_test_accuracy_lin=[0.14129810453762207*100,0.7880528431935669*100,0.7886272257323378*100]
office_test_accuracy_lin= [0.32222860425043076*100,0.9172889144170018*100,0.9172889144170018*100]
school1_test_accuracy_lin=[0.28948879954049395*100,0.8460654796094199*100,0.8460654796094199*100]
school2_test_accuracy_lin=[0.13038483630097644*100,0.7989661114302126*100,0.7978173463526709*100]

# plots
f, ax = plt.subplots(2, sharey=True)
# validation accuracy
ax[0].plot(C_range,court_valid_accuracy_lin)
ax[0].plot(C_range,office_valid_accuracy_lin)
ax[0].plot(C_range,school1_valid_accuracy_lin)
ax[0].plot(C_range,school2_valid_accuracy_lin)
ax[0].set_title('Linear Kernel - Validation Accuracy')
ax[0].grid(True)
ax[0].legend(['Courthouse', 'Office Building', 'School 1', 'School 2'], loc='best')
ax[0].set_ylabel('Accuracy (%)')
ax[0].set_xlabel('C')
# test accuracy
ax[1].plot(C_range,court_test_accuracy_lin)
ax[1].plot(C_range,office_test_accuracy_lin)
ax[1].plot(C_range,school1_test_accuracy_lin)
ax[1].plot(C_range,school2_test_accuracy_lin)
ax[1].set_title('Linear Kernel - Test Accuracy')
ax[1].grid(True)
plt.legend(['Courthouse', 'Office Building', 'School 1', 'School 2'], loc='best')
plt.ylabel('Accuracy (%)')
plt.xlabel('C')


# poly kernel function
C_range_poly=[0.0001, 0.001, 0.01]
court_valid_accuracy_poly=[0.12399540757749714*100,0.2835820895522388*100,0.6423650975889782*100]
court_test_accuracy_poly=[0.1556576680068926*100,0.30786904078116023*100,0.6611143021252154*100]

office_valid_accuracy_poly=[0.3668197474167623*100,0.7416762342135477*100,0.936280137772675*100]
office_test_accuracy_poly= [0.3543940264215968*100,0.7501435956346927*100,0.9345203905801264*100]

school1_valid_accuracy_poly=[0.2772675086107922*100,0.6142365097588978*100, 0.8151549942594719*100]
school1_test_accuracy_poly=[0.27627800114876505*100,0.6088454910970706*100,0.8299827685238369*100]

school2_valid_accuracy_poly=[0.15729047072330654*100,0.42709529276693453*100,0.7319173363949484*100]
school2_test_accuracy_poly=[0.15795519816197587*100,0.4152785755313039*100,0.7501435956346927*100]

# plots
f, a = plt.subplots(2, sharey=True)
# validation accuracy
a[0].plot(C_range_poly,court_valid_accuracy_poly)
a[0].plot(C_range_poly,office_valid_accuracy_poly)
a[0].plot(C_range_poly,school1_valid_accuracy_poly)
a[0].plot(C_range_poly,school2_valid_accuracy_poly)
a[0].set_title('Polynomial Kernel - Validation Accuracy')
a[0].grid(True)
a[0].legend(['Courthouse', 'Office Building', 'School 1', 'School 2'], loc='best')
a[0].set_ylabel('Accuracy (%)')
a[0].set_xlabel('C')
# test accuracy
a[1].plot(C_range_poly,court_test_accuracy_poly)
a[1].plot(C_range_poly,office_test_accuracy_poly)
a[1].plot(C_range_poly,school1_test_accuracy_poly)
a[1].plot(C_range_poly,school2_test_accuracy_poly)
a[1].set_title('Polynomial Kernel - Test Accuracy')
a[1].grid(True)
plt.legend(['Courthouse', 'Office Building', 'School 1', 'School 2'], loc='best')
plt.ylabel('Accuracy (%)')
plt.xlabel('C')

#plt.show()