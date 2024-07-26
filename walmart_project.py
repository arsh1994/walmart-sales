import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('Walmart.csv')
# print(data)

# data.info()


#checking null and duplicate values
# print(data.isnull().sum())
# duplicate = data[data.duplicated()]
# print(duplicate)

# checking stats of the data
data_stats = data.describe()
# print(data_stats)

# Outlier detection

import plotly.express as px

df_columns=data.select_dtypes(exclude = ['object'])

# for x in df_columns.columns:
#     ax=plt.boxplot(data[x])
#     ax=plt.title(x)
#     plt.show()

outliers_columns=['Weekly_Sales','Unemployment']    

for x in outliers_columns:

    Q1 = data[x].quantile(0.25)
  
    Q3 = data[x].quantile(0.75)
  
    IQR = Q3 - Q1
  
    LF = Q1 - 1.5 * IQR
  
    UF = Q3 + 1.5 * IQR
  
    data = data[(data[x] >= LF) & (data[x] <= UF)]

# print(data.shape)

# DOING EXPLORATORY DATA ANALYSIS

# sns.barplot(data=data,x=data['Holiday_Flag'],y=data['Weekly_Sales'])
# plt.show()

def inflat_range(cpi):
    if (cpi<=150):
        return "Low Inflation"
    if (cpi>=151 and cpi<=180):
        return "Mild Inflation"
    if (cpi>=181 and cpi<=200):
        return "Slighly High Inflation "
    else :
        return 'High Inflation'
    

new_column=[]
for i in range(0,len(data)):
    new_column.append(i)

data['inflation_range']=new_column  
data['inflation_range']=data['CPI'].apply(inflat_range)      
# print(data[data['inflation_range']=='Mild Inflation'])

# sns.barplot(data=data,x=data['inflation_range'],y=data['Weekly_Sales'])
# plt.show()
# insights
#AS WE CAN SEE THE SALES ARE HIGHER DURING LOW INFLATION AND IT IS LOW DURING HIGH INFLATION
#SUGGESTION -GIVE DISCOUNTS ON LOW PURCHASE VALUES TO ATTRACT CUSTOMERS BECAUSE PEOPLE GENERALLY DO NOT DO HIGH AMOUNT OF PUCHASING AT THE TIME OF HIGH INFLATION



#store wise sales

# sns.barplot(data=data,x=data['Store'],y=data['Weekly_Sales'])
# plt.show()

# store-4 and store-20 have the highest sales in comparison with other stores whereas store-33 has the lowest sales amongs all

# temp_wise_sales=data[['Weekly_Sales','Temperature']]
# print(temp_wise_sales)


# sns.scatterplot(data=data,x=data['Temperature'],y=data['Weekly_Sales'])
# plt.show()

data['Date'] = pd.to_datetime(data['Date'],dayfirst=True)
data['Year'] = data.Date.dt.year
data['Month'] = data.Date.dt.month
data['Day'] = data.Date.dt.day
data['Quarter'] = data.Date.dt.quarter
data['WeekOfYear'] = (data.Date.dt.isocalendar().week) * 1.0

print(data[data['Year']==2011])
      
# year_wise_sales=data.groupby(data['Year'])['Weekly_Sales'].sum().reset_index()      
# print(year_wise_sales)


#YEAR WISE SALES
# sns.barplot(data=year_wise_sales,x=year_wise_sales['Year'],y=year_wise_sales['Weekly_Sales'])
# plt.show()
# YEAR 2011 HAVE SEEN THE HIGHEST SALES WHEREAS YEAR 2012 IS ON THE LOWEST SIDE 
 
# sns.scatterplot(data=data,x=data['Date'],y=data['Weekly_Sales'])
# plt.show()

#SALES BY QUARTER

# sns.barplot(data=data,x=data['Quarter'],y=data['Weekly_Sales'])
# plt.show()

#ALL THE QUARTERS HAVE WITNESSED THE SIMILAR SALES

# MONTH WISE SALES:--

# sns.barplot(data=data,x=data['Month'],y=data['Weekly_Sales'],hue=data['Year'])
# plt.show()

# IT CAN BE SEEN FROM THE GRAPH THAT NO INFORMATION IS GIVEN FOR MONTH OF NOVEMBER AND DECEMBER FOR THE YEAR 2012
# AND BECAUSE THE MAJOR HOLIDAYS FALLS IN THE MONTH OF NOVEMBER AND DECEMBER ,THE SALES ARE HIGHER  
# IT ALSO SHOWS THAT JANAUARY MONTH HAVE THE LOWEST SALES DURING THE YEAR

# YEAR WISE WEEK OVER WEEK SALES
data['WeekOfYear']=data['WeekOfYear'].astype('int')

# sns.lineplot(data=data,x=data['WeekOfYear'],y=data['Weekly_Sales'],hue='Year')
# plt.show()

#IT IS ALSO EVIDENT IN THE WEEK OVER WEEK SALES THAT DURING HOLIDAY WEEK OF THE LAST MONTH WE CAN SEE THAT SALES ARE HIGHER DURING THAT PERIOD
# WHICH MEANS THAT DURING HOLIDAY PERIOD PEOPLE LIKE TO DO SHOPPING
#IT CAN ALSO BE SEEN THAT SALES FALLS IMMEDIATELY AFTER HOLIDAYS WEEKS

#
# sns.barplot(data=data,x=data['Holiday_Flag'],y=data['Weekly_Sales'],hue=data['Year'])
# plt.show()

#ON THE DAY OF HOLIDAY THE SALES ARE LITTLE BIT HIGHER

#IMPACT OF UNEMPLOYMENT RATE ON SALES
# sns.scatterplot(data=data,x=data['Unemployment'],y=data['Weekly_Sales'],hue=data['Weekly_Sales'])
# plt.show()

#IT CAN BE NOTICED THAT AT UNEMPLOYMENT RATE GREATER THAN 9 ,THE SALES IS DECREASING

#IMPACT OF FUEL PRICES ON SALES

# def fuel_range(price):
#     if (price<3):
#         return "Low"
#     if (price>=3 and price<=4):
#         return "Mid"
#     if (price>4):
#         return " High "
  
# fuel_wise_sales=data[['Fuel_Price','Weekly_Sales','CPI']]
# fuel_wise_sales['price_level']=data['Fuel_Price'].apply(fuel_range)
# print(fuel_wise_sales)

# sns.scatterplot(data=fuel_wise_sales,x=fuel_wise_sales['Fuel_Price'],y=fuel_wise_sales['Weekly_Sales'],hue=fuel_wise_sales['price_level'])
# plt.show()

#AS WE CAN SEE THAT FUEL PRICE IMPACTS GREATLY AND THE GRAPH IS SHOWING THAT SALES ARE HIGHER WHEN THE PRICE OF THE FUEL IS RANGES BETWEEN 3 TO 4
# AND SALES ARE DECREASING AT THE TIME WHEN FUEL PRICE IS GREATER THAN 4

#


# Drawing the heatmap to check relationship between variables

sns.heatmap(data=data.select_dtypes(include=int and float).corr(), cmap="YlGnBu", annot=True)
plt.show()

#PEARSON'S CORRELATION TELLS THE RELATIONSHIP BETWEEN RANGE(-1 TO +1) WHEN THE CORRELATION IS CLOSER TO ONE ,THE HIGHER THE RELATION BETWEEN VARIABLES AND VISE VERSA
#HERE WE CAN SEE THAT CPI AND UMEMPLOYMENT ARE NEGATIVELY CORRELATED WHICH MEANS THAT  WHEN CPI INCREASES UMEMPLOYMENT DECREASES AND SAME THING WE CAN SAY ABOUT WEEKLY SALES AND CPI

data1 = data.copy()
data1.set_index('Date', inplace=True)
store4 = data1
# there are about 45 different stores in this dataset.
sales4 = pd.DataFrame(store4.Weekly_Sales.groupby(store4.index).sum())
sales4.dtypes
sales4.head(20)
# Grouped weekly sales by store 4

#remove date from index to change its dtype because it clearly isnt acceptable.
sales4.reset_index(inplace = True)
sales4.set_index('Date',inplace = True)

sarimax_forecast=sales4.copy()

deseasonal_data=sales4-sales4.shift(6)
deseasonal_data=deseasonal_data.dropna()
print(sales4)
sales4.plot()
plt.show()
print(deseasonal_data)
deseasonal_data.plot()
plt.show()


from statsmodels.tsa.stattools import adfuller

# adfuller_test=adfuller(sales4['Weekly_Sales'])
adfuller_test=adfuller(deseasonal_data['Weekly_Sales'])
print(adfuller_test)
#null hyp=data is stationary p_val<0.05 accept null hypothesis
#alternative hyp= data is not stationary

p_val=adfuller_test[1]
print(p_val)#accept alternative hypothesis

from statsmodels.tsa.seasonal import seasonal_decompose
# d_trend.set_index('Date',inplace=True)
# d_trend.sort_index(inplace=True)
# print(d_trend)
analysis=deseasonal_data[['Weekly_Sales']].copy()
decompose_result=seasonal_decompose(analysis,period=12)
decompose_result.plot()
plt.show()

# sales4.plot()
# plt.show()

# print(sales4.shift())


# decomposition = seasonal_decompose(d_trend, period=12)  
# fig = plt.figure()  
# fig = decomposition.plot()  
# fig.set_size_inches(12, 10)
# plt.show()

from pmdarima import auto_arima
order=auto_arima(deseasonal_data['Weekly_Sales'],start_p=0,start_q=0,d=1,max_P=40,max_q=40)
print(order.summary())

train=deseasonal_data.iloc[:120]['Weekly_Sales'] 
test=deseasonal_data.iloc[120:]['Weekly_Sales']

tr=sarimax_forecast.iloc[:120]['Weekly_Sales']
tst=sarimax_forecast.iloc[120:]['Weekly_Sales']

# print(len(train))
# print(len(train)+len(test))

# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


from statsmodels.tsa.statespace.sarimax import SARIMAX
model=SARIMAX(train,order=(2,1,1),seasonal_order=(2,1,2,52))
model2=SARIMAX(tr,order=(2,1,1),seasonal_order=(2,1,2,52))

model=model.fit()
model2=model2.fit()

deseasonal_data['Predict']=model.predict(start=len(train),end=len(train)+len(test)-1)
deseasonal_data[['Weekly_Sales','Predict']].plot()
plt.show()

sarimax_forecast['Predict']=model2.predict(start=len(train),end=len(train)+len(test)-1)
sarimax_forecast[['Weekly_Sales','Predict']].plot()
plt.show()

forecast=model.forecast(steps=30)
predicted=pd.DataFrame(forecast)
print(predicted)
print(forecast)
plt.show()

forecast2=model2.forecast(steps=30)
predicted2=pd.DataFrame(forecast2)
print(predicted2)
print(forecast2)
plt.show()

from sklearn.metrics import mean_absolute_percentage_error

mape_sarimax=mean_absolute_percentage_error(predicted.iloc[:15],test.iloc[:15])
print('mape sarimax=',mape_sarimax)

mape_sarimax2=mean_absolute_percentage_error(predicted2.iloc[:15],tst.iloc[:15])
print('mape sarimax on orignal=',mape_sarimax2)


from prophet import Prophet

model=Prophet()
model=model.add_country_holidays(country_name='US')

fb_prof_pred=sales4.copy()
fb_prof_pred.reset_index(inplace=True)
fb_prof_pred.rename(columns={'Weekly_Sales':'y','Date':'ds'},inplace=True)
print(fb_prof_pred)

train_set=fb_prof_pred.iloc[:120]
test_set=fb_prof_pred.iloc[120:]

print(train_set)
print(test_set)

model.fit(train_set)
future=model.make_future_dataframe(periods=35,freq='W')
forecast=model.predict(future)

# m_2 = Prophet()
# m_2.add_country_holidays(country_name='IN')
# m_2.fit(df)


x_test_forecast=model.predict(test_set)
x_test_forecast.reset_index()
print(x_test_forecast)
print(forecast)

ax=plt.subplot()
test_set.plot(kind='line',x='ds',y='y',ax=ax)
x_test_forecast.plot(kind='line',x='ds',y='yhat',ax=ax)
plt.show()

from sklearn.metrics import mean_absolute_percentage_error
fb_prophet_mape=mean_absolute_percentage_error(test_set['y'],x_test_forecast['yhat'])
print('FBPROPHET MAPE = ',fb_prophet_mape)


#AS WE CAN SEE THAT MEAN ABSOLUTE PERCENTAGE ERROR OF SARIMAX IS THE LOWEST SO WE ARE SELECTING THE SARIMAX MODEL OF(2,1,1) ORDER



















