"""
ORIE 4741 - Final Project
Add new features
Grace Mitchell
12/2/2017

Purpose of script is to generate new csv files with new feature vectors for
all 4 facilties (courthouse, office building, school 1, school 2)
y = hourly electricity demand
"""
import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# 8712 rows
# starts at 7/1/16
# ends at 6/30/17
# 12/8 and 12/9 were omitted
# 24 points per day, 363 days total

# data frame
w=pd.DataFrame.from_csv('wfeatures.csv')

# Time of Day Feature
# Night 12:51AM -3:51AM --> 4 periods
# Early Morning 4:51AM -6:51AM -> 4 periods
# Morning 7:51AM -10:51AM -> 4 periods
# Afternoon 11:51AM -3:51PM -> 5 periods
# Evening 4:51PM -7:51PM -> 3 periods
# Night 8:51PM -11:51PM -> 4 periods
# create empty string array, add for day 1
tod=["" for x in range(24)]
for i in range(4):
    tod[i]='Night'
for i in range(4,8):
    tod[i]='Early Morning'
for i in range(8,12):
    tod[i]='Morning'
for i in range(12,17):
    tod[i]='Afternoon'
for i in range(17,20):
    tod[i]='Evening'
for i in range(20,24):
    tod[i]='Night'
# repeat sequence 363 times
tod_new=np.tile(tod,363)

# one hot encoding of TOD feature
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(tod_new)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# convert to list
onehot_encoded=onehot_encoded.tolist()
# empty arrays
TOD1=np.zeros(len(onehot_encoded))
TOD2=np.zeros(len(onehot_encoded))
TOD3=np.zeros(len(onehot_encoded))
TOD4=np.zeros(len(onehot_encoded))
TOD5=np.zeros(len(onehot_encoded))
# convert each column to a new list
for i in range(len(onehot_encoded)):
    TOD1[i]=onehot_encoded[i][0]
for i in range(len(onehot_encoded)):
    TOD2[i]=onehot_encoded[i][1]
for i in range(len(onehot_encoded)):
    TOD3[i]=onehot_encoded[i][2]
for i in range(len(onehot_encoded)):
    TOD4[i]=onehot_encoded[i][3]
for i in range(len(onehot_encoded)):
    TOD5[i]=onehot_encoded[i][4]

# Season Feature
# Summer 7/1/16 - 9/15/16 = 77 days
# Fall 9/16/16 - 11/15/16 = 61 days
# Winter 11/16/16 - 2/28/17 = 105 days
# Spring 3/1/17 - 5/15/17 = 75 days
# Summer 5/16/17 - 6/30/17 = 45 days
# create empty string array
season=["" for x in range(8712)]
for i in range(77*24):
    season[i]='Summer'
for i in range(77*24,138*24):
    season[i]='Fall'
for i in range(138*24,243*24):
    season[i]='Winter'
for i in range(243*24,318*24):
    season[i]='Spring'
for i in range(318*24,363*24):
    season[i]='Summer'

# one hot encoding of season feature
# integer encode
label_encoder1 = LabelEncoder()
integer_encoded1 = label_encoder1.fit_transform(season)
# binary encode
onehot_encoder1 = OneHotEncoder(sparse=False)
integer_encoded1 = integer_encoded1.reshape(len(integer_encoded1), 1)
onehot_encoded1 = onehot_encoder1.fit_transform(integer_encoded1)
# convert to list
onehot_encoded1=onehot_encoded1.tolist()
# empty arrays
season1=np.zeros(len(onehot_encoded1))
season2=np.zeros(len(onehot_encoded1))
season3=np.zeros(len(onehot_encoded1))
season4=np.zeros(len(onehot_encoded1))
# convert each column to a new list
for i in range(len(onehot_encoded1)):
    season1[i]=onehot_encoded1[i][0]
for i in range(len(onehot_encoded1)):
    season2[i]=onehot_encoded1[i][1]
for i in range(len(onehot_encoded1)):
    season3[i]=onehot_encoded1[i][2]
for i in range(len(onehot_encoded1)):
    season4[i]=onehot_encoded1[i][3]

# Weekend/Weekday Feature
# dataset starts on a Friday --> [Weekday, Weekend, Weekend, Weekday, Weekday, Weekday, Weekday]
# create empty string array
day=["" for x in range(7*24)]
for i in range(24):
    day[i]='Weekday'
for i in range(24,3*24):
    day[i]='Weekend'
for i in range(3*24,7*24):
    day[i]='Weekday'
# repeat sequence 51 times
day=np.tile(day,51) # 8568 data points
# add remaining days, 8712-8568 = 144 more points
# need 144/24 = 6 more days
day1=["" for x in range(24)]
for i in range(24):
    day1[i]='Weekday'
day23=["" for x in range(24*2)]
for i in range(24*2):
    day23[i]='Weekend'
day456=["" for x in range(24*3)]
for i in range(24*3):
    day456[i]='Weekday'
# concatenate
dayall=np.concatenate((day,day1,day23,day456),axis=0)

# one hot encoding of day type feature
# integer encode
label_encoder2 = LabelEncoder()
integer_encoded2 = label_encoder2.fit_transform(dayall)
# binary encode
onehot_encoder2 = OneHotEncoder(sparse=False)
integer_encoded2 = integer_encoded2.reshape(len(integer_encoded2), 1)
onehot_encoded2 = onehot_encoder1.fit_transform(integer_encoded2)
# convert to list
onehot_encoded2=onehot_encoded2.tolist()
# empty arrays
dayf1=np.zeros(len(onehot_encoded2))
dayf2=np.zeros(len(onehot_encoded2))
# convert each column to a new list
for i in range(len(onehot_encoded2)):
    dayf1[i]=onehot_encoded2[i][0]
for i in range(len(onehot_encoded2)):
    dayf2[i]=onehot_encoded2[i][1]

# Hour Index Feature
hi=np.linspace(1,24,num=24)
# repeat sequence 363 times
hi_new=np.tile(hi,363)
hi_new=hi_new.tolist()
# standardize
hi_s=[]
for i in range(len(hi_new)):
    hi_s.append((hi_new[i]-np.mean(hi_new))/np.std(hi_new))

# normalization for weather features
# create array for each feature
drytemp=w['HOURLYDRYBULBTEMPF'].tolist()
dewtemp=w['HOURLYDewPointTempF'].tolist()
humidity=w['HOURLYRelativeHumidity'].tolist()
wind=w['HOURLYWindSpeed'].tolist()
precip=w['HOURLYPrecip'].tolist()
# create offset
offset=np.ones(len(w),dtype=np.int)
# standardize features
drytemp_s=[]
for i in range(len(drytemp)):
    drytemp_s.append((drytemp[i]-np.mean(drytemp))/np.std(drytemp))
dewtemp_s=[]
for i in range(len(dewtemp)):
    dewtemp_s.append((dewtemp[i]-np.mean(dewtemp))/np.std(dewtemp))
humidity_s=[]
for i in range(len(humidity)):
    humidity_s.append((humidity[i]-np.mean(humidity))/np.std(humidity))
wind_s=[]
for i in range(len(wind)):
    wind_s.append((wind[i]-np.mean(wind))/np.std(wind))
precip_s=[]
for i in range(len(precip)):
    precip_s.append((precip[i]-np.mean(precip))/np.std(precip))
# create new data frame
wdf=pd.DataFrame({'offset':offset,'drytemp':drytemp_s,'dewtemp':dewtemp_s,'humidity':humidity_s,'wind speed':wind_s,'precip':precip_s,'TOD1':TOD1,'TOD2':TOD2,'TOD3':TOD3,'TOD4':TOD4,'TOD5':TOD5,'Season1':season1,'Season2':season2,'Season3':season3,'Season4':season4,'Day1':dayf1,'Day2':dayf2, 'Hour Index':hi_s})                                                   

# export non-specific features
#wdf.to_csv('nonspeffeatures.csv')

# Previous Hour Demand Feature
# courthouse

# create data frame
court=pd.DataFrame.from_csv('courthouse_1hr.csv')
# create array of electricity demand
court_y=court['kW'].tolist()
# convert from list to array
court_y = np.asarray(court_y)
courty = np.zeros(8712)
for i in range(8712):
    courty[i]=court_y[i-1]
courty[0]=court_y[0]
# standardize demand
# create new list
court_y_stan=[]
for i in range(len(court_y)):
    court_y_stan.append((court_y[i]-np.mean(court_y))/np.std(court_y)) 
# create new list
courty_stan=[]
for i in range(len(courty)):
    courty_stan.append((courty[i]-np.mean(courty))/np.std(courty))
# create new data frame
courtdf=pd.DataFrame({'PDemand Feature':courty_stan,'Courthouse y':court_y_stan})
# export 
#courtdf.to_csv('courtnewf_o.csv')

# office building

# create data frame
office=pd.DataFrame.from_csv('office building_1hr.csv')
# create array of electricity demand
office_y=office['kW'].tolist()
# convert from list to array
office_y = np.asarray(office_y)
officey = np.zeros(8712)
for i in range(8712):
    officey[i]=office_y[i-1]
officey[0]=office_y[0]
# standardize demand
# create new list
office_y_stan=[]
for i in range(len(office_y)):
    office_y_stan.append((office_y[i]-np.mean(office_y))/np.std(office_y)) 
# create new list
officey_stan=[]
for i in range(len(officey)):
    officey_stan.append((officey[i]-np.mean(officey))/np.std(officey))
# create new data frame
officedf=pd.DataFrame({'PDemand Feature':officey_stan,'Office y':office_y_stan})
# export 
#officedf.to_csv('officenewf_o.csv')
    
# school 1

# create data frame
school1=pd.DataFrame.from_csv('school 1_1hr.csv')
# create array of electricity demand
school1_y=school1['kW'].tolist()
# convert from list to array
school1_y = np.asarray(school1_y)
school1y = np.zeros(8712)
for i in range(8712):
    school1y[i]=school1_y[i-1]
school1y[0]=school1_y[0]
# standardize demand
# create new list
school1_y_stan=[]
for i in range(len(school1_y)):
    school1_y_stan.append((school1_y[i]-np.mean(school1_y))/np.std(school1_y)) 
# create new list
school1y_stan=[]
for i in range(len(school1y)):
    school1y_stan.append((school1y[i]-np.mean(school1y))/np.std(school1y))
# create new data frame
school1df=pd.DataFrame({'PDemand Feature':school1y_stan,'School1 y':school1_y_stan})
# export 
#school1df.to_csv('school1newf_o.csv')
    
# school 2

# create data frame
school2=pd.DataFrame.from_csv('school 2_1hr.csv')
# create array of electricity demand
school2_y=school2['kW'].tolist()
# convert from list to array
school2_y = np.asarray(school2_y)
school2y = np.zeros(8712)
for i in range(8712):
    school2y[i]=school2_y[i-1]
school2y[0]=school2_y[0]
# standardize demand
# create new list
school2_y_stan=[]
for i in range(len(school2_y)):
    school2_y_stan.append((school2_y[i]-np.mean(school2_y))/np.std(school2_y)) 
# create new list
school2y_stan=[]
for i in range(len(school2y)):
    school2y_stan.append((school2y[i]-np.mean(school2y))/np.std(school2y))
# create new data frame
school2df=pd.DataFrame({'PDemand Feature':school2y_stan,'School2 y':school2_y_stan})
# export 
#school2df.to_csv('school2newf_o.csv')

# shuffle and split to training and test: 5227 goes to training/1743 to validation/1742 to test

# courthouse
# create data frame
court_full=pd.DataFrame.from_csv('courtnewf_o.csv')
# randomize 
court_rand=court_full.sample(frac=1)
# split
court_train=court_rand[:5227]
court_valid=court_rand[5228:6970]
court_test=court_rand[6971:]
# export to csv
court_train.to_csv('courtnewf_train.csv')
court_test.to_csv('courtnewf_test.csv')
court_valid.to_csv('courtnewf_valid.csv')

# office
# create data frame
office_full=pd.DataFrame.from_csv('officenewf_o.csv')
# randomize 
office_rand=office_full.sample(frac=1)
# split
office_train=court_rand[:5227]
office_valid=court_rand[5228:6970]
office_test=court_rand[6971:]
# export to csv
office_test.to_csv('officenewf_test.csv')
office_train.to_csv('officenewf_train.csv')
office_valid.to_csv('officenewf_valid.csv')

# school 1
# create data frame
school1_full=pd.DataFrame.from_csv('school1newf_o.csv')
# randomize 
school1_rand=school1_full.sample(frac=1)
# split
school1_train=court_rand[:5227]
school1_valid=court_rand[5228:6970]
school1_test=court_rand[6971:]
# export to csv
school1_test.to_csv('school1newf_test.csv')
school1_train.to_csv('school1newf_train.csv')
school1_valid.to_csv('school1newf_valid.csv')

# school 2
# create data frame
school2_full=pd.DataFrame.from_csv('school2newf_o.csv')
# randomize 
school2_rand=school2_full.sample(frac=1)
# split
school2_train=court_rand[:5227]
school2_valid=court_rand[5228:6970]
school2_test=court_rand[6971:]
# export to csv
school2_test.to_csv('school2newf_test.csv')
school2_train.to_csv('school2newf_train.csv')
school2_valid.to_csv('school2newf_valid.csv')