# Mortgage_Bank_Loan_Analtsics
Mortgage Bank Loan Analytics with ARIMA and Machine Learning
Mortgage Loans Analytics
Banks can now use mortgage loan analytics using Data Science techniques. The system can provide detail information of the mortgage loans and the mortgage loan markets. It is a powerful tool for mortgage brokers to seek counterparties and generate trading interests and is useful for the CFOs to conduct what-ifs scenarios on the balance sheets.

Fill in the follow fields in Loan file template: 
•	Loan ID: to identify the special loan
•	Loan Type: to indicate the loan if fixed rate, or balloon loan , or ARM, or AMP (alternative mortgage product).
•	Balance:
•	Loan program type: to indicate conforming loan, FHA/VA loan, Jumbo loan or sub-prime loan
•	Current coupon rate:
•	Amortization type: the original amortization term
•	Maturity: the maturity loan (the remaining term of the loan)
•	FICO Score: the updated fico score
•	LTV: the current loan to value ratio
•	Loan Size: the loan amount of the loan
•	Loan origination location (City & Zip)
•	Unit Types (Types of property)

1.	Predicting mortgage demand using machine learning techniques
It is difficult for the financial institutions to determine the amount of personnel needed to handle the mortgage applications coming in. There are multiple factors influencing the amount of mortgage applications, such as the mortgage interest rates, which cause the amount of mortgage applications to differ day by day. In this research we aim to provide more insight in the amount of personnel needed by developing a machine learning model that predicts the amount of mortgage applications coming in per day for the next week, using the CRISP-DM framework. After conducting a literature study and interviews, multiple features are generated using historical data from a Dutch financial institution and external data. A number of machine learning models are developed and validated using cross-validation. The predictions of our best model differ on average mortgage applications per day compared to the actual amount of mortgage applications. A dynamic dashboard solution is proposed to visualize the predictions, in which mortgage interest rate changes can be manually entered in the dashboard, and recommendations have been given for the deployment of the model at the financial institutions.
Methodology

Historical data and publicly available data were used as input for our predictive model, and five machine learning techniques (Decision Tree, Random Forest, Support Vector Machines, Support Vector Regression and KNN ) were applied to create the predictions. The models are validated using repeated cross-validation, and evaluated using several evaluation criteria. We have also used ARIMA model, Linear Regression and Logistic Regression for predictive modeling.
Results:
The Random Forest model gave the best result on each of the four evaluation criteria used to evaluate the models. The Random Forest model is mortgage applications per day perform best, then Decision Tree model. The SVR model scored is the worse, SVM on a second place. The percent error of the Random Forest model is around of the actual amount of mortgage applications per day.
Linear Regression, 
Logistic Regression
Random Forests (RF) 
Support Vector Regression (SVR)
Support Vector Machine (SVM)
k-Nearest Neighbors
Decision Tree Classifier 
Using Scikit-learn, optimization of decision tree classifier performed by only pre-pruning. Maximum depth of the tree can be used as a control variable for pre-pruning. In the following the example, we can plot a decision tree on the same data with max_depth=4. Other than pre-pruning parameters, We have also tried other attribute selection measure such as entropy This pruned model is less complex, explainable, and easy to understand than the previous decision tree model plot.
Pros
Decision trees are easy to interpret and visualize.
It can easily capture Non-linear patterns.
It requires fewer data preprocessing from the user, for example, there is no need to normalize columns.
It can be used for feature engineering such as predicting missing values, suitable for variable selection.
The decision tree has no assumptions about distribution because of the non-parametric nature of the algorithm. 
Cons
Sensitive to noisy data. It can overfit noisy data.
The small variation(or variance) in data can result in the different decision tree. 
Decision trees are biased with imbalance dataset, so we can balance out the dataset before creating the decision tree.

2.	Decision Tree Algorithm
A decision tree is a flowchart-like tree structure where an internal node represents feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome. The topmost node in a decision tree is known as the root node. It learns to partition on the basis of the attribute value. It partitions the tree in recursively manner call recursive partitioning. This flowchart-like structure helps you in decision making. It's visualization like a flowchart diagram which easily mimics the human level thinking. That is why decision trees are easy to understand and interpret.

A predictive model was created using the SVM, KNN, Random Forest, and Deaccession Tree technique, which predicts the amount of mortgage applications per day(LoanInMonth)  with a mean absolute error of  mortgage applications per day. This can directly be converted to the amount of personnel needed at the mortgage application department of the financial institutions, by dividing it by the amount of mortgage applications handled per person per day. Based on mortgage per months, company can manage core human resources, such as Loan Processors, Mortgage Loan Originators, Underwriters, secondary  market analyst, lock desk personal,  compliance personals, & others. 

The mortgage interest rates have the biggest impact on our model, but are difficult to predict.
Several features (10 Years US Treasury Rate, Home Supply Index) can be added to the model in order to improve its predictive power. Furthermore, there is still improvement in the feature regarding mortgage interest rate changes, as a significant part of the error of our model is caused by under-prediction of the outliers.

3.1.1 Business Understanding
The background information is collected on the domain. A theoretical framework is developed by conducting a literature review which contains an overview of the related research in this research area, and an overview of the concepts in predictive analytics and the different models that are feasible for our Capstone project. For the domain analysis, background information on the mortgage application process and the mortgage domain is collected in order to get a better understanding of the different topics.
3.1.2 Data Understanding
In the Data Understanding stage the raw data is collected from the database using an SQL query and its characteristics and distributions are explored. Event logs are kept in the database and can be used for predictive modeling. Once the data is collected and saved as csv format, the data is explored and visualizations are made of the different variables to get an understanding of the data. With these visualizations we can already see some of the relationships in the data and identify possible features. The data exploration activity is important for becoming familiar with the data and identifying data quality problems.
3.1.3 Data Preparation
The goal of the Data Preparation stage is to transform and enrich the dataset so that it can be fed into
the models. After the data is collected and explored, it can be pre-processed so that it can be used
directly in our predictive model. With the pre-processed data one can perform feature engineering.
Using historical data and external data, different features can be generated. After the feature engineering activity, a subset of features will be selected that provide predictive value for our models.
3.1.4 Modeling
In this stage, several models are developed based on the dataset. First, a selection of predictive models is made (e.g. Linear Regression, Random Forests). These models are trained on the dataset and used tomake predictions. The models are validated using a test set and repeated 5-fold cross-validation. 
For each of the models, hyperparameters are optimized and data pre-processing is done if needed (e.g. centering, scaling, multi-collinearity checks). Some of the models have specific requirements on the form of the data, which require specific pre-processing activities  
4.2.2 Mortgage interest rates
Mortgage interest rates have a significant impact on the amount of mortgage applications. If the interest rates are low, the mortgages are relatively cheaper for the borrower as they have to pay less interest, which leads to an increased amount of mortgage applications. A high mortgage interest rate means the mortgage borrower pays a high amount of interest to the lender, which makes the mortgage less attractive for the borrower. Interest rate changes have a significant impact on mortgage applications, as was seen in November of last year, where a sudden increase in interest rates led to a large peak in mortgage applications.  The main difference between the mortgages offered by these types of companies lies in the mortgage interest rates. Even a small difference in mortgage interest rates can often save or cost the borrower a vast amount of money, due to the large sum of a mortgage.  
In general, there are two types of mortgage interest rate: variable rates (ARM) and fixed rates. Variable interest rates are generally lower than fixed interest rates, but can change every month. Fixed interest rates are slightly higher, but are fixed for a certain period of time. A fixed interest rate is generally preferred when the mortgage interest rates are expected to rise, or when the borrower wants to know its monthly expenses upfront. A variable interest rate (ARM) is preferred when interest rates are expected to decrease. If a financial institution has a significantly higher interest rate than its competitors, it will generally receive fewer mortgage applications as the independent mortgage advisors will forward its customers to a different mortgage lender.
For the financial institutions, there can be a number of reasons to change its mortgage interest rate.
First of all, the mortgage interest rate is based on the cost of lending for the financial institutions itself. 
If the cost of debt is higher, the financial institutions will compensate this by charging higher interest rates for its mortgages, in order to keep a profitable margin on their products. This cost of lending is mainly based on the capital market interest rate, for the long-term loans, and the short-term loans. If either of these changes significantly, one can expect the financial institutions to respond by changing their own mortgage interest rates. This usually happens after a few days. 

Second, financial institutions generally work with a budget for their mortgages. Based on the amount of
funding they can get, and on the interest rates and the duration of the funding, they determine a budget
for their mortgages for the upcoming period. Ideally, financial institutions want to match the duration
of the fixed interest period of a mortgage with the duration of the lending of debt for that mortgage.
Once a financial institution is almost out of budget for a specific fixed interest period, it may choose to
increase the interest rate for mortgages with that fixed interest period. This way, borrowers will apply
for mortgages with a different fixed interest period, or may choose to go to another mortgage lender.

Finally, financial institutions sometimes increase their interest rates during the summer months, and at
the end of the year, as there is less personnel available to handle the requests due to vacations and
holidays. With less personnel available they can handle less mortgage requests, so in order to keep the
processing time the same they choose to reduce the input, by increasing the interest rates. Financial
institutions may also specifically keep interest rates low for mortgages with a certain fixed interest
period. Interest rate changes are not always directly influenced by changes in the cost of lending, but can have numerous reasons.
US10Y.index = pd.to_datetime(US10Y.index)
US10Y.resample('M',how=mean).plot(title="Interest Rate for 10 Years Treasury - ")
plt.xlabel('Loan origination Date')
plt.xticks(rotation=60)
plt.ylabel('US 10 Year Treasury Rates')
plt.title('Mortgage Bank: US 10 Year Treasury Rates')
monthly_rate=US10Y.resample('M',how=mean)
type(monthly_rate)
monthly_rate_data=monthly_rate['RATE']
type(monthly_rate_data)
type(monthly_loan_rev_data)

Generally, when US 10 Years Treasury Rate fluctuates, that leads lenders to adjust their internal bank rates accordingly. Also interest rates for consumers varies on several risk factors, such as DTI (Debt to Income Ratio), FICO Scores, Recent derogatory events on their credit history, stable job history, W2 or 1099, Stated Income, Profit or Loss Statements, Student Loans, Auto Payments, Credit utilization, Property types, number of households, rental history, etc.
 
Interest Rate is currently historical low. In the short run rate may go ups and down but in the long run rate will go up. As housing price goes up, interest rate will go up to control the housing price.

Now we will compare Monthly Revenue, Monthly Closed Loan Number and Active Mortgage Loan Originators. We will count number of MLO actively closing loans on any given month.

mlo_num=data[['Loan Officer Name']]
mlo_num['date'] = pd.DatetimeIndex(data['Created Date'])
mlo_num = mlo_num.set_index('date')
monthly_mlo_num=mlo_num.resample('M').nunique()
from matplotlib.lines import Line2D
colors = ['red', 'green', 'blue']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
labels = ['Monthly closed loans', 'Monthly Loan Revenue / 400000', 'Monthly Active MLO * 4']
plt.plot(monthly_loan_num_data, color='red')
plt.plot(monthly_loan_rev_data/400000, color='green')
plt.plot(monthly_mlo_num*4, color='blue')
plt.legend(lines, labels)
plt.title('Comparing Monthly: Revenue, Closed_Loan_Num, Active_MLO_Num')
plt.show()

 
As we know that number of producer is essential component in any given business. MLO (Mortgage loan Originator) is core component in Mortgage business. Many MLO works indedendently and interect directly to clients, involve in marketting and grow their business. There could be many MLO in Mortgage Bank, but active MLO generate more reverue for the bank. As number of active MLO goes up, which will directly and positively impact numbers of loan closed per month, eventually mortgage revenue will go up. On the other hand, once number of active MLO goes down, mortgage revenue and number of loan per month goes down as well. By visualizing the graphs, we can see that monthly data of closed loan numbers , monthly revenue and active MLO numbers ber months, all moving at the same direction.

Let’s find out interest rate effect on Monthly Closed Loans and Monthly Revenue.

from matplotlib.lines import Line2D
colors = ['red', 'green', 'blue']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
labels = ['Monthly closed loans', 'Monthly Loan Revenue / 400000', '10 Years Interest Rate * 20']
plt.plot(monthly_loan_num_data, color='red')
plt.plot(monthly_loan_rev_data/400000, color='green')
plt.plot(monthly_rate_data*20, color='blue')
plt.legend(lines, labels)
plt.title('Comparing Monthly Revenue, Monthly Closed Loans VS US 10 Year Treasury Rates')
plt.show()
r_monthly_loan_num_data_monthly_loan_rev = pearson_r(monthly_loan_num_data, monthly_loan_rev_data)
print('Pearson correlation coefficient between Monthly Closed Loans & Monthly loan Rev: ', r_monthly_loan_num_data_monthly_loan_rev)
print('Pearson correlation coefficient between Monthly Interest & Monthly loans Closed Data: ', r_monthly_Rate_monthly_loan_data )

Pearson correlation coefficient between Monthly Closed Loans & Monthly loan Rev:  0.8295841987344892
Pearson correlation coefficient between Monthly Interest & Monthly loans Closed Data:  -0.3343475255276081

 
We can see the strong positive correlation between Monthly Closed Loans and Monthly Revenue. This graph also suggest that, as interest rates goes down, banks monthly revenue and numbers of loans increases, and when the Rates goes up, both Monthly Closed Loans and Monthly Revenue for the Mortgage bank decline. Pearson correlation coefficient between Monthly Interest & Monthly loans Closed Data is  -0.334, which clearly proves that Monthly Interest Rates & Monthly loans Closed Data is negatively correlated.

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_cltv_fico = draw_bs_pairs(
        cltv_data, fico, pearson_r, size=1000)
bs_replicates_loan_num_loan_rev = draw_bs_pairs(
        monthly_loan_num_data, monthly_loan_rev_data, pearson_r, size=1000)
# Compute 95% confidence intervals
conf_int_cltv_fico = np.percentile(bs_replicates_cltv_fico, [2.5, 97.5])
conf_int_loan_num_loan_rev = np.percentile(bs_replicates_loan_num_loan_rev, [2.5, 97.5])
print('CLTV data VS Qualification FICO Data       :', r_cltv_data_fico, conf_int_cltv_fico)
print('Monthly Loan_num_data VS. Monthly_loan_rev :', r_monthly_loan_num_data_monthly_loan_rev, conf_int_loan_num_loan_rev)

Acquire 1000 pairs bootstrap replicates of the Pearson correlation coefficient using the draw_bs_pairs() function you wrote in the previous exercise for CLTV data VS Qualification FICO Data and Monthly Loan_num_data VS. Monthly_loan_rev. Compute the 95% confidence interval for both using your bootstrap replicates. We have created a NumPy array of percentiles to compute. These are the 2.5th, and 97.5th. By creating a list and convert the list to a NumPy array using np.array(). For example, np.array([2.5, 97.5]) would create an array consisting of the 2.5th and 97.5th percentiles.

CLTV data VS Qualification FICO Data       : -0.037060429592168175 [-0.08  0.  ]
Monthly Loan_num_data VS. Monthly_loan_rev : 0.8295841987344892 [0.73 0.9 ]

''' It shows that there is statistically significant relationship between number of loans closed and loan revenue'''
Random Walk
Are Interest Rates or Monthly Loan Returns Prices a Random Walk?
Most returns prices follow a random walk (perhaps with a drift). We will look at a time series of Monthly Sales Revenue, and run the 'Augmented Dickey-Fuller Test' from the statsmodels library to show that it does indeed follow a random walk. With the ADF test, the "null hypothesis" (the hypothesis that we either reject or fail to reject) is that the series follows a random walk. Therefore, a low p-value (say less than 5%) means we can reject the null hypothesis that the series is a random walk.
Print out just the p-value of the test (adfuller_loan_rev_data[0] is the test statistic, and adfuller_loan_rev_data[1] is the p-value). Print out the entire output, which includes the test statistic, the p-values, and the critical values for tests with 1%, 10%, and 5% levels.
# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller
monthly_loan_rev_data
# Run the ADF test on the monthly_loan_rev_data series and print out the results
adfuller_loan_rev_data = adfuller(monthly_loan_rev_data)
print(adfuller_loan_rev_data)
print('The p-value of the test on loan_rev is: ' + str(adfuller_loan_rev_data[1]))

(-4.49967715626648, 0.00019690419763896495, 10, 40, {'1%': -3.6055648906249997, '5%': -2.937069375, '10%': -2.606985625}, 1307.285137861741)
The p-value of the test on loan_rev is: 0.00019690419763896495

'''According to this test, p-value is very low (lower than 0.05). We reject the hypothesis that monthly_loan_rev_data follow a random walk. '''
 
Let’s try same for Monthly Loan Data:

# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller
monthly_loan_num_data
# Run the ADF test on the monthly_loan_num_data series and print out the results
adfuller_loan_num_data = adfuller(monthly_loan_num_data)
print(adfuller_loan_num_data)
print('The p-value of the test on loan_num is: ' + str(adfuller_loan_num_data[1]))

(-5.7659481612690495, 5.532460937310067e-07, 0, 50, {'1%': -3.568485864, '5%': -2.92135992, '10%': -2.5986616}, 300.2408550906095)
The p-value of the test on loan_num is: 5.532460937310067e-07

According to this test, p-value is very low (lower than 0.05). We reject the hypothesis that monthly_loan_num_data follow a random walk. 
Let’s try same for Interest Rate Data:

monthly_rate_data=monthly_rate['RATE']
type(monthly_rate_data)
# Run the ADF test on the monthly_rate_data series and print out the results
adfuller_monthly_rate_data = adfuller(monthly_rate_data)
print(adfuller_monthly_rate_data)
print('The p-value of the test on monthly_rate_data is: ' + str(adfuller_monthly_rate_data[1]))

(-1.396544767435691, 0.5839314748568241, 1, 49, {'1%': -3.5714715250448363, '5%': -2.922629480573571, '10%': -2.5993358475635153}, -34.97121939430423)
The p-value of the test on monthly_rate_data is: 0.5839314748568241

According to this test, p-value is very is higher than 0.05. We cannot reject the hypothesis that Monthly Interest Rate prices follow a random walk. 
Are Interest Rates Auto correlated?
When we look at daily changes in interest rates, the autocorrelation is close to zero. However, if we resample the data and look at monthly or annual changes, the autocorrelation is negative. This implies that while short term changes in interest rates may be uncorrelated, long term changes in interest rates are negatively auto correlated. A daily move up or down in interest rates is unlikely to tell us anything about interest rates tomorrow, but a move in interest rates over a year can tell us something about where interest rates are going over the next year. And this makes some economic sense: over long horizons, when interest rates go up, the economy tends to slow down, which consequently causes interest rates to fall, and vice versa. If we want to look at the data by monthly and annually. We can easily resample and sum it up. I’m using ‘M’ as the period for resampling which means the data should be resampled on a month boundary and 'A' for annual data'. Finally find the autocorrelation of annual interest rate changes.

US10Y['change_rates'] = US10Y.diff()
US10Y['change_rates'] = US10Y['change_rates'].dropna()
# Compute and print the autocorrelation of daily changes
autocorrelation_daily = US10Y['change_rates'].autocorr()
print("The autocorrelation of daily interest rate changes is %4.2f" %(autocorrelation_daily))

The autocorrelation of daily interest rate changes is -0.06

US10Y.index = pd.to_datetime(US10Y.index)
annual_rate_data = US10Y['RATE'].resample(rule='A').last()
# Repeat above for annual data
annual_rate_data['diff_rates'] = annual_rate_data.diff()
annual_rate_data['diff_rates'] = annual_rate_data['diff_rates'].dropna()
print(annual_rate_data['diff_rates'])
autocorrelation_annual = annual_rate_data['diff_rates'].autocorr()
print("The autocorrelation of annual interest rate changes is %4.2f" %(autocorrelation_annual))
DATE
2015-12-31    0.10
2016-12-31    0.18
2017-12-31   -0.05
2018-12-31    0.37
Freq: A-DEC, Name: RATE, dtype: float64
The autocorrelation of annual interest rate changes is -0.97

'''Notice how the daily autocorrelation is small but the annual autocorrelation is large and negative'''
4.2.3 Changes in regulations
Another factor that impacts the amount of mortgage applications is changes in regulations. Depending on the type of regulations change and the impact of the change, there is generally an increase or decrease in mortgage applications before and after the regulations change. Changes in the
mortgage loan regulations include amongst others changes in the maximum mortgage, also called the Loan-To-Value (LTV) ratio, and changes in the mortgage interest deduction. The LTV ratio is a financial term that indicates the ratio of the mortgage loan to the value of the property.
4.2.4 Other predictors
Besides the mortgage interest rates and changes in regulations, several other predictors were
mentioned in the literature and interviews. These will be discussed briefly below. Furthermore, besides
looking at the predictors of the amount of mortgage applications, we will also look at factors that affect
the housing market. Since there is a strong relationship between the housing market and the mortgage
market (i.e. the amount of houses sold and the amount of mortgage applications), we can assume that
the factors that influence the housing market may affect the amount of mortgage applications. The amount of mortgage applications are low in Winter and there are peaks in May, June, July.  We also expect a drop in the amount of mortgage applications during the weekends, as most of the mortgage advisors are not working during these days. Furthermore, holidays are expected to have a negative 
impact on the amount of mortgage applications. Finally, even though there seems to be no autocorrelation between the amount of mortgage applications over time, we still want to include the historical amount of mortgage applications in our model. Even though the amount of mortgage applications are not dependent on the amount of mortgage applications for the previous day, week or month, there is still a time component present and there may be a correlation between these factors.

First of all, the average house prices and expected changes in house prices have an effect on the amount of mortgage applications. If house prices are expected to increase, people might think it is a good moment to buy a house, and thus more mortgage applications may be coming in at the financial institutions. House prices do not only influence the amount of mortgage applications, but mortgage applications in return also influence house prices. 

Second, rental prices are important. If the cost of renting is high, buying becomes more attractive compared to renting. Third, the amount of houses available has an effect on the amount of mortgages sold. If the supply of houses is large, the average house prices are going to drop and it will become more attractive to buy a house. All of these factors influence the amount of houses sold and thus the amount of mortgage applications. In the interviews each of these factors was identified as a possible predictor. 
The economic features may also impact the amount of mortgage applications, but this is generally caused by the media. If there are a relatively large number of news stories about economic growth or an economic crisis within a short period of time, then this will have a certain impact on the consumers.
4.3 PREDICTIVE ANALYTICS
Predictive analytics is a field in data mining that encompasses different statistical and machine learning
techniques that are aimed at making empirical predictions. These predictions are based on empirical data, rather than predictions that are based on theory only. In predictive analytics, several statistical and machine learning techniques can be used to create predictive models. These models are used to exploit patterns in historical data, and make use of these patterns in order to predict future events. These models can be validated using different methods to determine the quality of such a model, in order to see which model performs best. 

There are generally two types of problems predictive analytics is used for: classification problems and
regression problems. The main difference between these two problems is the dependent variable, the
target variable that is being predicted. In classification problems, the dependent variable is categorical
(e.g. credit status). In regression problems, the dependent variable is continuous (e.g. pricing).
The techniques that are used in predictive analytics to create a model depend heavily on the type of
problem. For classification problems, classification techniques are used such Random Forest and
decision trees. These techniques often consist out of one or multiple algorithms that can be used to
construct a model. For decision trees, some of the algorithms are Classification and Regression Trees. 

For regression problems, regression techniques such as multiple linear regression, support vector
machines or time series are used. These techniques focus on providing a mathematical equation in order to represent the interdependencies between the independent variables and the dependent variable, and use these to make predictions. One of the most popular regression techniques is linear regression. When applied correctly, regression is a powerful technique to show the relationships between the independent and the dependent variables. However, linear regression requires some assumptions in the dataset. One of these assumptions is that there has to be a linear interdependency between the independent variables and the dependent variable. A pitfall of linear regression is that the regression line contains no information about the distribution of the data. It needs to be combined with a visualization of the regression line in order to draw conclusions.

It can be seen that different datasets that have the same means, variances, correlation and linear fit, still have a completely different distribution, even though their regression lines are the same. Hence, a regression line always needs to be combined with a visualization in order to draw conclusions about the distribution of the data.

In order to compensate for the disadvantages of the individual models, ensemble models can be used.
An ensemble model is a set of individually trained models, which predictors are combined to increase
the predictive performance. Ensemble models are generally more accurate than any of the individual
models that make up the ensemble model.

Examples of techniques used for creating an ensemble model are bagging and boosting. With bagging, multiple versions of a predictor are used to create an aggregated predictor, in order to increase the accuracy of the model. An example of a bagging algorithm is random forests, which combines a set of decision trees to increase the model performance. A combination of the machine learning techniques mentioned above can be used to create predictive models. These models can then be validated and compared based on predictive power, which can be calculated using a set of statistical measures, such as Mean Absolute Error (MAE), Root Mean Square Error (RMSE) and R2

5 DATA UNDERSTANDING
The Data Understanding stage has been split up in two parts: data collection and data exploration. In the data collection we will discuss characteristics of the event log data, and how the data has been extracted from the database. In the data exploration, the data and its characteristics are explored to extract useful information for our models.
5.1 DATA COLLECTION
Since we are only interested in the event log data we will only be using one of the tables. This table contains data about every mortgage application. Every action performed by the system or by a user on a mortgage application is logged, and the status before and after that specific action is logged. For our analysis we are mainly interested in the date and time at which each of the mortgage applications have entered the system.
Besides Mortgage Application DataSet, we have joined two separate (10 Years US Treasury Rate, Home Supply Index) with our existing Mortgage Application DataSet to enhance predictive power of our model.
US10Y.replace('.', -99999, inplace=True)
US10Y= US10Y.replace(to_replace=[-99999], value=[np.nan])
US10Y= US10Y.fillna(method='ffill')
US10Y= US10Y[['RATE']].astype('float64')
total_missing_rate=US10Y.isnull().sum()
model_data1['Loan Amount'].pct_change()
US10Y['RATE'].pct_change()
US10Y['RATE'].autocorr()
US10Y['RATE'].diff()

#Join two DataFrames model_data1 & US10Y save the results in model_data

model_data2 = model_data1.join(US10Y)
total_missing=model_data2.isnull().sum()
model_data2= model_data2.fillna(method='ffill')
total_missing=model_data2.isnull().sum()

#Integrate Monthly housing supply index data and merging with current dataset

total_missing_home_supply=home_supply.isnull().sum()
model_data = model_data2.join(home_supply)
total_missing=model_data.isnull().sum()
model_data= model_data.fillna(method='ffill')
total_missing=model_data.isnull().sum()
5.2 DATA EXPLORATION
Since our dataset can be grouped per day to create meaningful visualizations. The dataset contains data from October 2014 until December 2018. In order to get a feel of the amount of mortgage applications per day and the distribution of the mortgage applications, different visualizations can be made using Python. Two graphs have been created, which can be found in Figure 1 and Figure 2. Both of these graphs only contain the amount of mortgage applications on the weekdays. As there are almost no applications coming in on the weekends they have been excluded from the graphs.
As can be seen from the graphs, there seems to be a seasonal pattern on a monthly level, but from these graphs it is not very clear. It also seems like there are some outliers, so these data points will have to be investigated to see if they will have to be included in our model, as there can be multiple underlying reasons for outliers in our dataset. It also seems there is an increase in mortgage applications during the last few months of each year. The amount of applications per day during these months is higher compared to the other months. This can have multiple explanations so this will have to be accounted for in the model.

The density plot shows the distribution of the amount of mortgage applications. It seems the distribution of the amount of mortgage applications is normally distributed, slightly skewed to the right with a long tail. This is due to the outliers mentioned before. The median seems to be at around mortgage applications per day.

Let’s plot number of Loan Application for Connecticut:
#Plot loan amount per Statae
plt.xlabel('Loan origination Data')
plt.xticks(rotation=60)
plt.ylabel('Loan Amount')
plt.title('MWB Loan Data for CT')
plt.plot(ct_data['Created Date'], ct_data['Loan Amount'])

 
We can see a spike of new loan application on summer 2018 for state of Connecticut. 

We can analyze mortgage loan origination for the bank per state; we also can find the total counts of different types of loan for the bank. Based on this information bank can allocate resources:

print('====================================================')
print('=========    Total Sales by State     ==============')
print(' ')
print('Total Sales in Cunnecticut   : $', ct_loan_amount)
print('Total Sales in Florida       : $', fl_loan_amount)
print('Total Sales in New York      : $', ny_loan_amount)
print('Total Sales in New Jersey    : $', nj_loan_amount)
print('Total Sales in Pennsylvania  : $', pa_loan_amount)
print(' ')
print('====================================================')
print(' ')
#####################################
loan_types=data['Loan Type'].unique()
group_loan_types=data.groupby(data['Loan Type']).size()
print('Unique Loan Types        : ', loan_types)
print(' ')
print('====================================================')
print('Number of loan per Types : ', group_loan_types)

=========    Total Sales by State     ==============
 
Total Sales in Connecticut   	: $ 3604641.0
Total Sales in Florida       	: $ 3371495.0
Total Sales in New York      	: $ 868836552.28
Total Sales in New Jersey    	: $ 236456261.06
Total Sales in Pennsylvania  	: $ 600920.0
 
====================================================
 
Unique Loan Types        :  ['Residential' 'FHA' 'Commercial' 'Conventional' 'Other' 'VA']
 
====================================================
Number of loan per Types :  Loan Type
Commercial       			100
Conventional    			1736
FHA              			418
Other             			31
Residential      			215
VA                 			1

Let’s visualize the loan data per state;
from matplotlib.ticker import FuncFormatter
x = np.arange(5)
money = [1.5e5, 2.5e6, 5.5e6, 1.0e7, 2.0e7, 3.0e7, 4.0e7, 5.0e7, 6.0e7]
def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)
formatter = FuncFormatter(millions)
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.bar(St, loan_amount_per_state)
plt.xticks(x, ('Connecticut', 'Florida', 'New Jersey', 'New York', 'Pennsylvania'))
plt.ylabel('Loan Amount')
plt.xlabel('Loan Origination per State')
plt.xticks(rotation=60)
plt.title('Mortgage Bank Loans per State')
plt.show()
 
We can see that New York and New Jersey is the major Loan Origination market for the Bank.
Let’s explore Loan Origination Data with Unit Types:

total_unit_type = data.groupby(data['Unit Type']).size()
print('Loan originated in all States per unit types : \n', total_unit_type)
==============================
 Loan originated in all States per unit types : 
 Unit Type
Condo           	216
Coop             	76
FourFamily       	36
Industrial        	1
Land              	1
MixedUse        	25
MultiFamily      	20
OneFamily      	1276
PUD               	6
ThreeFamily     	129
TwoFamily       	707
Warehouse        8
We can see that One Family and Two Family property dominates the Loan Origination shares for the mortgage bank. As a result company should focus on suitable loan products particularly for 1-4 family loan products. This Mortgage Bank also originates good number of Condo and Coop.
We can analyze data for individual MLO (Mortgage Loan Originator)

formatter = FuncFormatter(millions)
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.xlabel('Loan origination Date')
plt.xticks(rotation=60)
plt.ylabel('Loan Amount')
plt.title('Mortgage Loan Data for MLO(1b5 Ch4wdh5ry)')
plt.plot(abu_loan['Created Date'], abu_loan['Loan Amount'])
plt.xlabel('Loan origination Date')
plt.xticks(rotation=60)
plt.ylabel('Loan Amount')
plt.title('Mortgage Loan Data for MLO(1b5 Ch4wdh5ry)')
plt.bar(abu_loan['Created Date'], abu_loan['Loan Amount'])
 
We can easily explore the Loan Origination history for particular MLO

Here is total Sales volume per MLO.  
formatter = FuncFormatter(millions)
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.xlabel('Loan Officer')
plt.ylabel('Loan Amount')
plt.title('Mortgage Loan Data per MLO')
plt.xticks(rotation=90)
plt.plot(sales_totals_lo)
 
Some MLO generates major portions of the sales revenue for the Bank, many of the MLO is performing way below company standards. Additional research required for the root causes behind the performance. Let’s explore the Loan Statistics data for the Mortgage Bank:

**********************  Loan Statistics for Mortgage Bank ***********************
Average Loan Amount is              	:  $ 445099.91
Median Loan Amount is               	:  $ 400000.00
Standard deviation of is 		 :  $ 288803.96
Minimum Loan Amount is           	 :  $ 23600.00
Maximum Loan Amount is          	:  $ 6125000.00
Total of Loan Amount is              		 :  $ 1113194869.34
10% of Loan Amount is below              	:  $ 200000.00
25% of Loan Amount is below             	 :  $ 292500.00
50% of Loan Amount is below             	:  $ 417000.00
75% of Loan Amount is below             	:  $ 533000.00
90% of Loan Amount is below              	:  $ 696500.00

FICO Score is one of the critical information for processing mortgage loan application.
**************** Qualification FICO  Statistics for Mortgage Bank ***************
Average FICO is                      :  $ 727.81
Median FICO is                     :  $ 732.00
Standard deviation is            :  $ 56.61
Minimum FICO is                  :  $ 0.00
Maximum FICO is                  :  $ 825.00
 5% of FICO is below              :  $ 636.00
10% of FICO is below             :  $ 653.00
25% of FICO is below             :  $ 693.00
40% of FICO is below             :  $ 716.00
50% of FICO is below             :  $ 732.00
65% of FICO is below             :  $ 755.00
75% of FICO is below             :  $ 771.00
90% of FICO is below             :  $ 795.00
95% of FICO is below             :  $ 803.00
 ==========================================================================
We can see that 50% of the consumers FICO score is between 693 and 771. Let’s visualize FICO statistics 
fico_score = [fico_q7, fico_q0, fico_q1, fico_q5, fico_q2, fico_q6, fico_q3, fico_q4, fico_q8]
fico_pct=['5% Loan Below', '10% Loan Below', '25% Loan Below', '40% Loan Below', '50% Loan Below', '65% Loan Below', '75% Loan Below','90% Loan Below', '95% Loan Below']

plt.xticks(rotation=75)
plt.ylim((500,850))
plt.bar(fico_pct, fico_score)
plt.ylabel('Qualification FICO Scores')
plt.xlabel('Qualification FICO (%)')
plt.title('Mortgage Loans Qualification FICO  Statistics')
plt.show()
 
We can see that 90% of the consumers FICO score is over 650. Most of cases applicant with lower FICO score will not qualify for Conventional mortgages. In many cases, FHA Loan type could be only option remaining for the applicants with FICO score; many bank uses cut-off points for FICO Score (640-680) for conventional mortgages. FHA accepts FICO score below 600.

Let’s Analyze Mortgage Loan Officer’s Sales Volume per Loan Type. Processing different loan types need different expertise. Some loan officer do not have experience for VA or commercial loan at all. Few Loan officers are expert in commercial side of loan origination process. In order to Loan Origination for 1-4 units of residential properties, MLO must be licensed by the State. On the other hand other properties such as multi-family, mixed use or commercial loans do not require MLO to be licensed by the State.

print('**************** Mortgage Loan Ofiicer'' Sales Volume per Loan Type ***************')
lo_loan = data[['Loan Officer Name', 'Loan Type', 'Created Date', 'Loan Amount']]
#We can use groupby to organize the data by category and name.
lo_loan_group = lo_loan.groupby(['Loan Officer Name', 'Loan Type']).sum()
lo_loan_group_count = lo_loan.groupby(['Loan Officer Name', 'Loan Type']).count()

print(lo_loan_group)
print(lo_loan_group_count)
lo_loan['Loan Type'].describe()
'''The category representation looks good but we need to break it apart 
to graph it as a stacked bar graph. unstack can do this for us.'''
lo_loan_group.unstack().head()
lo_loan_group_plot = lo_loan_group.unstack().plot(kind='bar',stacked=True,title="Total Sales by Loan Officers by Loan Type",figsize=(9, 7))
lo_loan_group_plot.set_xlabel("Loan Officers")
lo_loan_group_plot.set_ylabel("Sales per Loan Type")
lo_loan_group_plot.legend(["Commercial","Conventional","FHA","Other","VA"], loc=2,ncol=3)
plt.ylim(10000000, 140000000)
 

Similarly, we can explore Loan Officer’s Sales Volume per Unit Type.
 

Now that we know who the biggest customers are and how they purchase products, 
we might want to look at purchase patterns in more detail. Let’s take another look at the data and try to see how large the individual purchases are. A histogram allows us to group purchases together so we can see how big the customer transactions are.

from IPython.display import display   # A notebook function to display more complex data (like tables)
import scipy.stats as stats  

import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
loan_patterns = data[['Loan Amount', 'Created Date']]
loan_patterns_plot = loan_patterns['Loan Amount'].hist(alpha=0.6, bins=40, grid=True)
loan_patterns_plot = loan_patterns['Loan Amount'].apply(np.sqrt)
param = stats.norm.fit(loan_patterns_plot) 
x = np.linspace(0, 100000, 1250000)      # Linear spacing of 100 elements between 0 and 20.
pdf_fitted = stats.norm.pdf(x, *param)    # Use the fitted paramters to 
loan_patterns_plot.plot.hist(alpha=0.6, bins=40, grid=True, density=True, legend=None)
plt.text(x=np.min(loan_patterns_plot), y=800, s=r"$\mu=%0.2f$" % param[0] + "\n" + r"$\sigma=%0.2f$" % param[1], color='b')
# Plot a line of the fitted distribution over the top
# Standard plot stuff
plt.xticks(rotation=75)
plt.ylim((1,900))
plt.xlim((1,1210000))
plt.xlabel("Loan Amount($)")
plt.ylabel("Loan frequency")
plt.title("Histogram with fitted normal distribution for Mortgage Bank Loan")
print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis(loan_patterns_plot) ))
print( 'skewness of normal distribution (should be 0): {}'.format( skew(loan_patterns_plot) ))
print("mean : ", np.mean(loan_patterns_plot))
print("var  : ", np.var(loan_patterns_plot))
print("skew : ",skew(loan_patterns_plot))
print("kurt : ",kurtosis(loan_patterns_plot))

 

excess kurtosis of normal distribution (should be 0): 10.20423896299732
skewness of normal distribution (should be 0): 1.6231675369974472
mean :  644.086535155362
var  :  30252.443004451427
skew :  1.6231675369974472
kurt :  10.20423896299732

The fancy text to show us what the parameters of the distribution are (mean and standard deviation). We can create a histogram with 20 bins to show the distribution of purchasing patterns. Fit a normal distribution to the data then plot the histogram. 

One of the really cool things that Pandas allows us to do is resample the data. We want to look at the data by month, we can easily resample and sum it all up. I’m using ‘M’ as the period for resampling which means the data should be resampled on a month boundary.

loan_patterns.resample('M',how=sum)
monthly_loan_rev=loan_patterns.resample('M',how=sum)
loan_patterns.info()
loan_patterns_month_plot = loan_patterns.resample('M',how=sum).plot(title="Mortgage Bank - Total Sales by Month",legend=True)
loan_patterns_month_plot.set_xlabel("Months")
loan_patterns_month_plot.set_ylabel("Monthly Slaes")
plt.xticks(rotation=45)
plt.ylim((12000000, 35000000))
fig = loan_patterns_month_plot.get_figure()

 
We can see that monthly mortgage loan sales volume varies between $15M and $32M. Another interesting find is Loan sales are at the peak during summer seasons. Winter sales are normally slow but 2018 was an exception.
data = pd.read_csv('c:/scripts/mwb2014.csv', index_col='Created Date', parse_dates=True, encoding='cp1252')
Best month for Sales
monthly_loan_amount_sum = by_month['Loan Amount'].sum()
daily_loan_amount_sum = by_day['Loan Amount'].sum()
yearly_loan_amount_sum = by_year['Loan Amount'].sum()
monthly_loan_amount_sum = monthly_loan_amount_sum
monthly_loan_amount_sum_sorted= monthly_loan_amount_sum.sort_values()
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.xlabel('Loan Origination Month')
plt.ylabel('Loan Amount')
plt.title('MWB : 24 Months Total - Monthly Loan Sales (Sorted)')
plt.xticks(rotation=90)
plt.plot(monthly_loan_amount_sum_sorted)
 

Monthly Worst Sales: January, April and September and Best is June, December and October.
Comparing year-over-year performance:
plt.plot(M17, color='red')
plt.plot(M18, color='blue')
red_patch = mpatches.Patch(color='red', label=('2017 Loans (Jan - Sep) Total: ', M17['Created Date'].sum()) ) 
blue_patch = mpatches.Patch(color='blue', label=('2018 Loans (Jan - Sep) Total: ',  M18['Created Date'].sum()) )
plt.legend(handles=[red_patch, blue_patch])
plt.xticks(range(len(M17)), [calendar.month_name[month] for month in M17.index], rotation=60)
plt.xlabel('Loan origination Months')
#plt.xticks(rotation=60)
plt.ylabel('Loans per Months')
plt.title(Mortgage Bank Monthly Loan numbers 2017 & 2018: ')
plt.show()
 
In 2017, Mortgage Bank has closed more loan compare to 2018
 
In 2017, Mortgage Bank sales revenue was (M_amount_17.sum()) =  $180M 
For 2018, (M_amount_18.sum())= $197M

monthly_loan_num.resample('M', how='count')
monthly_loan_num_data= monthly_loan_num.resample('M', how='count')
monthly_loan_num_plot = monthly_loan_num.resample('M',how='count').plot(title="Mortgage Bank - Total Sales by Month",legend=True)
plt.xlabel('Loan origination Months')
plt.xticks(rotation=60)
plt.ylabel('Loans per Months')
plt.title('Mortgage Bank Monthly Loan numbers')

 
r_monthly_loan_num_data_monthly_loan_rev = pearson_r(monthly_loan_num_data, monthly_loan_rev_data)
print('Pearson correlation coefficient between Monthly Closed Loans & Monthly loan Rev: ', r_monthly_loan_num_data_monthly_loan_rev)
Pearson correlation coefficient between Monthly Closed Loans & Monthly loan Rev:  0.8295841987344892 
Monthly Loan numbers and Monthly Sales volumes are strongly positively correlated. Once we analyze number of the loans closed per month, we find the similarity between sales volume and loan numbers. As average loan numbers goes up, total monthly sales volume goes up. Any given months, if the number of loans closed are higher; we find that average loan amounts are low for that particular months. In other words, this may suggest that loan processing requirements and guidelines loans with higher loan amount take longer time to close the loans with lower loan amounts.

'''Visual exploration is the most effective way to extract information between variables.

We can plot a barplot of the frequency distribution of a categorical feature using the seaborn package, which shows the frequency distribution of the mortgage dataset column. 
