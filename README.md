# Customer_Segmentation

## Overview

- Used recency, frequency and monetary (RFM) to analyse customer purchases from the United Kingdom
- Used elbow plot to determine suitable number of clusters
- Applied k-means clustering to segment the customers into 3 different groups:
  - Small spending customers
  - Middle spending customers
  - Large spending customers

## Tools Used

- Language: Python 
- Packages: numpy, pandas, seaborn, matplotlib, datetime, sklearn, mpl_toolkits
- Data: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online+retail#), download link [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx)
- Topics: Python, RFM, Clustering, K-means Clustering

## Data

The data is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. It consists of 541909 rows and 8 columns and its details are shown below:

| Variable    | Description                                                                                                                                                 |
|:------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------|
| InvoiceNo   | Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'C', it indicates a cancellation. |
| StockCode   | Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.                                                         |
| Description | Product (item) name. Nominal.                                                                                                                               |
| Quantity    | The quantities of each product (item) per transaction. Numeric.                                                                                             |
| InvoiceDate | Invoice Date and time. Numeric, the day and time when each transaction was generated.                                                                       |
| UnitPrice   | Unit price. Numeric, Product price per unit in sterling.                                                                                                    |
| CustomerID  | Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.                                                                     |
| Country     | Country name. Nominal, the name of the country where each customer resides.   

## Data Cleaning

The following was done to clean up the data:

- Filtered out purchases from the United Kingdom
- Removed null values from `CustomerID` and `Description` and negative values from `Quantity` and `UnitPrice`
- Filtered out `InvoiceNo` which starts with letter 'C'
- Converted `CustomerID` and `StockCode` to categorical, converted `InvoiceDate` to datetime object
- Dropped `Country` column

## Feature Engineering

The following 2 columns was created:

**InvoiceDay** - Day of the invoice

**TotalSum** - Total amount customer purchased for a particular item

## Exploratory Data Analysis

Most Purchased Items  |  
:-------------------------:|
![alt text](https://github.com/nchin212/Customer_Segmentation/blob/gh-pages/plots/bar1.png)


Number of Customers over Time  |  Amount Spent over Time
:-------------------------:|:-------------------------:
![alt text](https://github.com/nchin212/Customer_Segmentation/blob/gh-pages/plots/line1.png) |  ![alt text](https://github.com/nchin212/Customer_Segmentation/blob/gh-pages/plots/line2.png)



## RFM

Created 3 new columns, `Recency`, `Frequency` and `Monetary`. Their details are as follows:

**Recency** - The number of days since last purchase by the customer

**Frequency** - The total number of orders made by the customer

**Monetary** - The total amount spent by the customer

Applied log transformation to make the data normally distributed.

## Modelling

K-means clustering requires the number of clusters to be predefined. We used an elbow plot to determine the suitable number of clusters.

![alt text](https://github.com/nchin212/Customer_Segmentation/blob/gh-pages/plots/elbow.png)

Selected k=3 and clustered the customers into 3 different groups as visualised below:

![alt text](https://github.com/nchin212/Customer_Segmentation/blob/gh-pages/plots/cluster.png)

## Extracting the Clusters

Decided to separate the 3 different groups as follows:

- **Small spending customers**: High recency, low frequency, low monetary
- **Middle spending customers**: Middle recency, middle frequency, middle monetary
- **Large spending customers**: Low recency, High frequency, High monetary

## Relevant Links

**Jupyter Notebook :** https://nchin212.github.io/Customer_Segmentation/cust_seg.html

**Portfolio :** https://nchin212.github.io/post/customer_segmentation/
