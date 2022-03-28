#!/usr/bin/env python
# coding: utf-8

'''Explore online engagement data by state

Classes:

    state_analysis()

Functions:

    state_query () -> object
    by_district(object)
    engagment_2020(string, object, int)
    state_cloud(object, object, str, str):
    lunch_line(object, object):   
    lunch_box(object, object):
    black_his_line(object, object):
    black_his_box(object, object):
    product_outliers(object):
    demographic_values(object):
    engagement_month(object, int):
'''

# for data manipulation
import pandas as pd
import numpy as np

#for creating word clouds
import random
from collections import Counter
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator

# files for graph visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# for time series visualizations
from pandas.tseries.offsets import BDay
isBusinessDay = BDay().is_on_offset
from datetime import datetime

# self made modules
from sql_db import sql_fun
from ml_analysis import analysis

# to access main folder location
import os

import warnings
warnings.filterwarnings("ignore")


class state_analysis():
    ''' This is a class for analyzing online engagement by state.
      
    Attributes:
        state (string): The name of the state to analyse.
    '''
    
    def __init__(self, state = None): 
        ''' Initializes analysis object

        Parameters:
        state (string): The name of the state to analyse.
        host  (string): password for database
        user (string): username for database
        password(string): password for database
        database: name of database
        '''

        self.state = state
        self.host = "localhost"
        self.user = "root"
        self.password = "#Pandas&Penguins"
        self.database = "learn"


    #queries SQL database to get information related to product use and demographics for districts in the state
    def state_query(self):
        ''' Queries database to get information needed for analysis

            Returns:
            results of query in the form of a pandas dataframe
            '''
        query =   f'''SELECT 
                        d.district_id as district, 
                        state_districts as state, 
                        locale, 
                        black_hispanic_range AS black_his,
                        free_reduced_lunch_range AS free_reduced, 
                        expend_student_range AS expend, 
                        a.product_id,
                        date,
                        engagement_index AS engagement,
                        name
                    FROM 
                        districts d
                    JOIN 
                        product_access a 
                    ON d.district_id = a.district_id
                    JOIN products p
                    ON a.product_id = p.product_id
                    WHERE state_districts = "{self.state}"
                    ''' 
        s = sql_fun(self.host, self.user, self.password, self.database)  #create database connection             
        state_df = s.query(query)
        
        return state_df

    def by_district(self, state_df):  
        ''' Queries database to get information needed for analysis

            Parameters:
            state_df(str): dataframe to use for analysis
            ''' 
        # create a line plot of engagement index for each district to see if there are any major differences
        district_df = state_df.groupby(['district', 'date'])['engagement'].mean().reset_index()

        match_day = pd.to_datetime(district_df['date']).map(isBusinessDay) #exclude weekends from line plot
        district_df_week = district_df[match_day]

        g = sns.relplot(
            data=district_df_week,
            x = 'date', y = 'engagement', col = 'district', hue = 'district',
            kind = 'line', palette = 'crest', linewidth = 4, zorder = 5,
            col_wrap = 3, height = 2, aspect = 3, legend = False)

        # Iterate over each subplot to customize further
        for district, ax in g.axes_dict.items():

            # Add the title as an annotation within the plot
            ax.text(.8, .85, district, transform = ax.transAxes, fontweight = 'bold')

        ax.set_xticks(ax.get_xticks()[::2])

        # Tweak the supporting aspects of the plot
        g.set_titles('')
        g.set_axis_labels('', 'engagement index')
        g.tight_layout()


    def engagment_2020(self, state, state_df, scaler = 1):
        ''' Prints a line plot showing how online engagement changed over the year

            Parameters:
            state(str): name of state 
            state_df(str): dataframe to use for analysis
            scaler(int) amount to scale the engagement index down by (through division)
        ''' 
        # access processed_data folder and read california school status csv
        calendar_query = f'''SELECT
                            *
                        FROM
                        school_calendars
                        WHERE state_calendars = "{state}"
                        '''

        s = sql_fun(self.host, self.user, self.password, self.database) #create database connection 
        calendar_df =  s.query(calendar_query)
        calendar_df = calendar_df[['date', 'hybrid', 'inperson', 'virtual']]

        #create a line plot of engagement index over the year
        match_series = pd.to_datetime(state_df['date']).map(isBusinessDay) #exclude weekends from line plot
        state_df_week = state_df[match_series]
        state_df_week['engagement_scaled'] = state_df_week['engagement']/scaler

        fig, axes = plt.subplots(figsize = (20,7), dpi = 80)
        plt.title('Online Engagement and Mode of School Attendance', fontsize = 30, color = '#b00e0e')
        l = sns.lineplot(data = state_df_week, x = 'date', y = 'engagement_scaled') 
        calendar_df.plot(x='date', ax=l) 

    def state_cloud(self, state_df, img = None, color = 'red', contour = '#b00e0e', title = None):
        ''' Create a word cloud showing top ten products based on 
            average engagement from top ten products for each district

            Parameters:
            state_df(str): dataframe to use for analysis
            color(str): color for state outline
        ''' 
        # random color generator for word cloud
        # function below adapted from: https://stackoverflow.com/questions/43124136/change-the-font-color-of-generated-word-cloud
        def random_color_func(word, font_size, position, orientation, random_state = None,
                            **kwargs):
            num = random.randint(100, 255)  
            return f'hsl({num}, 100%%, %50d%%)' % random.randint(60, 100)  

        # group by district and product name, find average engagement by product name for each district
        engage_average = state_df.groupby(['district', 'name'])['engagement'].mean().reset_index()
        engage_average_sorted = engage_average.sort_values(by = ['district', 'engagement'], ascending = False)

        # use the sorted dataframe to get the ten most used products per district
        top_10_engagement = engage_average_sorted.groupby('district').head(10).reset_index(drop = True)

        # find the engagement average of all products in the top ten by name, create a dictionary for the word cloud
        top_ten_engagement = top_10_engagement.groupby(['name'])['engagement'].mean()
        sums_dict = top_ten_engagement.to_dict()

        # create an analysis object and a word cloud
        high = analysis()
        cloud = high.word_cloud(dictionary = sums_dict, image=img, contour = contour)
        
        # show the word cloud
        plt.figure(figsize=(60,30))
        plt.imshow(cloud.recolor(color_func=random_color_func, random_state = 3), interpolation = 'bilinear')
        if title != None:
            plt.title(title, fontsize = 50, color = color)
        plt.axis('off')
        plt.show()

    def lunch_line(self, state_df, replace = None):
        ''' Create a line plot showing online engagement
            by percent of students receiving free and reduced lunch

            Parameters:
            state_df(object): dataframe to use for analysis
            replace(object): dictionary of values to be replaced 
                and their replacement values.
        ''' 

        # if some categories have too few districts,
        # they can be combined by providing a replacement dictionary
        if replace != None:
            lunch = state_df.replace(replace)
        else:
                lunch = state_df

        # group data by free_reduced lunch category
        average_lunch = lunch.groupby(['free_reduced', 'date'])['engagement'].mean().reset_index()

        #exclude weekends from line plot
        match_day = pd.to_datetime(average_lunch['date']).map(isBusinessDay) 
        lunch_week = average_lunch[match_day]

        # plot data
        fig, axes = plt.subplots(figsize = (20,7), dpi = 80)
        l = sns.lineplot(data = lunch_week, x = 'date', y = 'engagement', hue = 'free_reduced')  

    def lunch_box(self, state_df, replace = None):
        ''' Create a box plot showing online engagement
            by percent of students receiving free and reduced lunch

            Parameters:
            state_df(object): dataframe to use for analysis
            replace(object): dictionary of values to be replaced 
                and their replacement values.
        ''' 

    # if some categories have too few districts, 
    # they can be combined by providing a replacement dictionary

        if replace != None:
            lunch = state_df.replace(replace)
        else:
            lunch = state_df
            
        # group data by free_reduced lunch category
        average_lunch = lunch.groupby(['free_reduced', 'date'])['engagement'].mean().reset_index()

        #exclude weekends from line plot
        match_day = pd.to_datetime(average_lunch['date']).map(isBusinessDay) 
        lunch_week = average_lunch[match_day]

        #create a box plot of engagement index, looking at free reduced lunch
        fig, axes = plt.subplots(figsize=(20,7), dpi= 80)
        lunch_week['month'] = [d.strftime('%b') for d in lunch_week.date] #create a month field to order graph by month

        y = sns.boxplot(x='month', y='engagement', hue = 'free_reduced', data=lunch_week, order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    def black_his_line(self, state_df, replace = None):
        ''' Create a line plot showing online engagement
            by percent of students in school who are black or hispanic

            Parameters:
            state_df(object): dataframe to use for analysis
            replace(object): dictionary of values to be replaced 
                and their replacement values.
        ''' 
        ## if some categories have too few districts,
        # they can be combined by providing a replacement dictionary
        if replace != None:
            black_his = state_df.replace(replace)
        else:
            black_his = state_df
        
        # group data by black_his category
        average_black_his = black_his .groupby(['black_his', 'date'])['engagement'].mean('').reset_index()

        #exclude weekends from line plot
        match_day = pd.to_datetime(average_black_his['date']).map(isBusinessDay) #exclude weekends from line plot
        black_his_week = average_black_his[match_day]

        #plot data
        fig, axes = plt.subplots(figsize = (20,7), dpi = 80)
        l = sns.lineplot(data = black_his_week, x = 'date', y = 'engagement', hue = 'black_his')  

    def black_his_box(self, state_df, replace = None):
        ''' Create a line plot showing online engagement
            by percent of students in school who are black or hispanic

            Parameters:
            state_df(object): dataframe to use for analysis
            replace(object): dictionary of values to be replaced 
        '''
        #if some categories have too few districts,
        # they can be combined by providing a replacement dictionary     
        if replace != None:
            black_his = state_df.replace(replace)
        else:
            black_his = state_df

        # group data by black_his category
        average_black_his = black_his .groupby(['black_his', 'date'])['engagement'].mean('').reset_index()

        #exclude weekends from plot
        match_day = pd.to_datetime(average_black_his['date']).map(isBusinessDay) #exclude weekends from line plot
        black_his_week = average_black_his[match_day]

        #create a box plot of engagement index, percentage of black and hispanic students
        black_his_week['month'] = [d.strftime('%b') for d in black_his_week.date] #create a month field to order graph by month

        fig, axes = plt.subplots(figsize=(20,7), dpi= 80)
        y = sns.boxplot(x='month', y='engagement', hue = 'black_his', data=black_his_week, order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    def product_outliers(self, state_df):
        ''' Print a dataframe sorted by engagement index to look for outliers

            Parameters:
            state_df(object): dataframe to use for analysis 
        '''
        # look at products with highest engagement index (look at outliers)
        print(state_df[['engagement', 'name', 'district', 'locale', 'black_his', 'free_reduced', 'expend']]
            .sort_values(by = ['engagement'], ascending = False).head(40).to_string(index = False))

    def demographic_values(self, state_df):
        ''' Print a dataframe showing demographics so plots can be modified if needed

            Parameters:
            state_df(object): dataframe to use for analysis 
        '''
        demographics_df = state_df[['district', 'locale', 'black_his', 'free_reduced', 'expend']].drop_duplicates()
        print(demographics_df)

    def engagement_month(self, state_df, outliers = None):
        ''' Print a box plot of engagement index to look for outliers

            Parameters:
            state_df(object): dataframe to use for analysis 
        '''
        
        dist_ave_engage = state_df.groupby(['district', 'date', 'free_reduced', 'black_his'])['engagement'].mean('').reset_index()
        dist_ave_engage['month'] = [d.strftime('%b') for d in dist_ave_engage.date]

        if outliers != None:
            dist_ave_engage['engagement'].values[dist_ave_engage['engagement'].values > outliers] = outliers
        
        # Draw Plot
        fig, axes = plt.subplots(figsize=(20,7), dpi= 80)
        t = sns.boxplot(x='month', y='engagement', data=dist_ave_engage, order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])