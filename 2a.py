import numpy as np
import tensorflow as tf
import pandas as pd
import datetime as dt
import sqlite3

meansurment_datetime = 'meansurment_datetime'
temperature = 'temperature'
humidity = 'humidity'
lowcost_pm2_5 = 'lowcost_pm2'


data1 = pd.read_csv('data/jeftini_senzori.TXT', sep=";", header=None)
data1.columns = ['node','date','hour','energy','temperature','humidity','NO2contectrations','PM1concentrations'
                 ,'PM2.5cocentrations','PM10concentrations','PM1std','PM2.5std','PM10std','latitude','longitude']


df1 = pd.DataFrame(columns=[meansurment_datetime,temperature,humidity,lowcost_pm2_5])
sum_column = data1['date'] + " " + data1['hour']

df1[meansurment_datetime] = sum_column
df1[meansurment_datetime] = pd.to_datetime(df1[meansurment_datetime], format="%Y/%m/%d %H:%M:%S")
df1[temperature] = data1['temperature']
df1[humidity] = data1['humidity']
df1[lowcost_pm2_5] = data1['PM2.5std']


date_beginning = 'date beginning'
time_beginning = 'time beginning'
date_end = 'date end'
time_end = 'time end'

data = pd.read_excel('data/skupi_senzori.xlsx', header=4, skipfooter=7)

amb = data.columns[len(data.columns)-3]
head,sep,tail = amb.partition(' ')
data.rename(columns={amb: head}, inplace=True)


df = pd.DataFrame(data)


i = 0
for val in df[date_beginning]:
    if(isinstance(val, dt.datetime)):
        date = dt.date.strftime(val.date(),'%m/%d/%Y')
        df[date_beginning][i] = date
    i+=1

i=0
for val in df[time_beginning]:
    time = dt.time.strftime(val,'%H:%M:%S')
    df[time_beginning][i] = time
    i+=1

i=0
for val in df[date_end]:
    if (isinstance(val, dt.datetime)):
        date = dt.date.strftime(val.date(), '%m/%d/%Y')
        df[date_end][i] = date
    i += 1

i=0
for val in df[time_end]:
    time = dt.time.strftime(val,'%H:%M:%S')
    df[time_end][i] = time
    i+=1

datetime_beginning = 'datetime_beginning'
datetime_end = 'datetime_end'
reference_pm2_5 = 'reference_pm2_5'
df2 = pd.DataFrame(columns=[datetime_beginning,datetime_end,reference_pm2_5])


df2[datetime_beginning] = df[date_beginning] + " " + df[time_beginning]
df2[datetime_end] = df[date_end] + " " + df[time_end]
df2[datetime_beginning] = pd.to_datetime(df2[datetime_beginning],format='%d/%m/%Y %H:%M:%S')
df2[datetime_end] = pd.to_datetime(df2[datetime_end],format='%d/%m/%Y %H:%M:%S')

df2[reference_pm2_5] = df['PM2.5_ambient']


df2[datetime_beginning] = df2[datetime_beginning] + pd.DateOffset(hours=-2)
df2[datetime_end] = df2[datetime_end] + pd.DateOffset(hours=-2)

datetime_reference_beginning = 'datetime_reference_beginning'
datetime_reference_end = 'datetime_reference_end'
reference_pm2_5 = 'reference_pm2.5'


conn = sqlite3.connect(':memory:')

df1.to_sql('df1',conn,index=False)
df2.to_sql('df2',conn,index=False)

qry = '''
    select
        meansurment_datetime meansurment_datetime,
        temperature temperature,
        humidity humidity,
        lowcost_pm2 lowcost_pm2_5,
        datetime_beginning datetime_reference_beginning,
        datetime_end datetime_reference_end,
        reference_pm2_5 reference_pm2_5
    from
        df2 join df1 on meansurment_datetime where meansurment_datetime between datetime_reference_beginning and datetime_reference_end
    '''

data = pd.read_sql_query(qry, conn)
data.to_excel('2a.xls')