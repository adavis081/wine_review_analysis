"""
The purpose of this script is to import the wine review data, clean data
as needed, while conducting some exploratory analysis to become familar with the data.
"""

# Import all needed packages
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize


# Import data
reviews = pd.read_csv("Data/winemag-data-130k-v2.csv")


# Look at the size of the dataset
print(len(reviews)) #129,971 rows
print(reviews.columns) # 14 columns


"""
To begin to clean this dataset, we will start looking at missing values.
"""


# Check counts of missing values
print(reviews.isna().sum())


# the country column is only missing 63 values, easiest to drop those rows
reviews_clean = reviews.dropna(subset = 'country')


# price has 8996 missing values, need to investigate distribution
print(reviews_clean['price'].describe())

# histogram of price
figsize(7, 5)
plt.hist(reviews_clean['price'], color='blue', edgecolor='black', bins= 50)
plt.xlabel('Price')
plt.ylabel('Price Frequency')
plt.title('Price Frequencies')

plt.savefig("Visualizations/hist_price_original.png")


# some very expensive wines are heavily skewing the distribution
len(reviews_clean[reviews_clean['price'] > 500])


# only 91 wines more than $500, lets remove them as outliers
reviews_price_less_than_500 = reviews_clean[reviews_clean['price'] <= 500]


# histogram of price < $500
figsize(7, 5)
plt.hist(reviews_price_less_than_500['price'], color='blue', edgecolor='black', bins= 50)
plt.xlabel('Price')
plt.ylabel('Price Frequency')
plt.title('Price Frequencies')

plt.savefig("Visualizations/hist_price_less_than_500.png")


# still heavily skewed, only take wines under $200
reviews_price_less_than_200 = reviews_clean[reviews_clean['price'] <= 200]


# histogram of price < $200
figsize(7, 5)
plt.hist(reviews_price_less_than_200['price'], color='blue', edgecolor='black', bins= 50)
plt.xlabel('Price')
plt.ylabel('Price Frequency')
plt.title('Price Frequencies') # still skewed but much better (shorter tails)

plt.savefig("Visualizations/hist_price_less_than_200.png")



# now we can fill in the missing values for price with the median, and only looks at review under $200
price_to_fill = int(reviews_price_less_than_200['price'].mean())
reviews_clean['price'].fillna(price_to_fill, inplace=True)



"""
Missing values for country, price, and variety have been taken care of. The remaining fields
(designation, region, taster_name, and taster_twitter_handle) with missing values probably do not 
need to be addressed. If they do, they can be added to the data cleaning script.

Outliers were also eliminated by removing entries where the price of the wine was extremley high.
"""


# save cleaned data
reviews_clean.to_csv("Data/reviews_clean.csv", index = False)





