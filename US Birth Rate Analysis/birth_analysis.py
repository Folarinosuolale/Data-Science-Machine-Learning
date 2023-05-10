#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('births.csv', delimiter=',')

#data preprocessing

# Filter out any rows where gender is not F or M
df = df[df['gender'].isin(['F', 'M'])]

# Replace null values in the births column with the average
mean_births = df['births'].mean()
df['births'].fillna(mean_births, inplace=True)

# Remove values outside of the normal range for day and month
df = df[(df['day'] >= 1) & (df['day'] <= 31)]
df = df[(df['month'] >= 1) & (df['month'] <= 12)]


# Visualization 1: Number of births by year and gender
# We can create a bar plot to visualize the number of births by year and gender.
#This will give us an idea of any trends that exist in the data:

births_by_year_gender = df.groupby(['year', 'gender']).sum()['births'].unstack()

v1 = births_by_year_gender.plot(kind='bar', figsize=(10, 6))
plt.title('Number of Births by Year and Gender')
plt.xlabel('Year')
plt.ylabel('Number of Births')
plt.legend(['Female', 'Male'])

# set y-axis ticks to actual values
yticks = v1.get_yticks()
v1.set_yticklabels([int(y) for y in yticks])
plt.show()


#Visualization 2: Distribution of births by month and gender
#Next, we can create a line plot to visualize the distribution of births by month and gender. 
#This will give us an idea of any seasonality that exists in the data:

# Group by month and gender and calculate the mean births for each group
births_by_month_gender = df.groupby(['month', 'gender']).mean()['births'].unstack()

# Create a line plot to visualize the mean number of births by month and gender
v2 = births_by_month_gender.plot(kind='line', figsize=(10, 6))
plt.title('Mean Number of Births by Month and Gender')
plt.xlabel('Month')
plt.ylabel('Mean Number of Births')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(['Female', 'Male'])
plt.show()

# set y-axis ticks to actual values
yticks = v2.get_yticks()
v2.set_yticklabels([int(y) for y in yticks])
plt.show()
plt.show()

#WE CAN ALSO USE A HISTOGRAM TO FURTHER VISUALIZE THIS
# Create separate dataframes for male and female births
df_male = df[df['gender'] == 'M']
df_female = df[df['gender'] == 'F']

# Create histograms for male and female births by month
plt.hist([df_male['month'], df_female['month']], bins=12, alpha=0.5, label=['Male', 'Female'])
plt.title('Distribution of Births by Month and Gender')
plt.xlabel('Month')
plt.ylabel('Number of Births')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.show()


#Visualization 3: Number of births by day of the week
#Finally, we can create a bar plot to visualize the number of births by day of the week.
#This will give us an idea of any patterns that exist in the data

births_by_day = df.groupby('day').sum()['births']

v3 = births_by_day.plot(kind='bar', figsize=(8, 6))
plt.title('Number of Births by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Births')

# set y-axis ticks to actual values
yticks = v3.get_yticks()
v3.set_yticklabels([int(y) for y in yticks])

plt.show()
