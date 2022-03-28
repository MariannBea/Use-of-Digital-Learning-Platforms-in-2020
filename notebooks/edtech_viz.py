#!/usr/bin/env python
# coding: utf-8

'''Creates visualizations for analysis.

Classes:

     visualize()

Functions:

   grade_bars(object, str, str, float, float, str, str, float, float, float)
   high_not_low(object, str, int) -> set
   low_scores(object, str, str)
   high_scores(object, str, str)
   make_cloud_dict(object, str, str)
   make_cloud_list(object, str, str)
   random_color_func(str, int, int, int)
   association_prepare(object, str, object)
   sort_rules(object) -> object, object, object, object
   segment_plot(object)  
'''

import os
import random

# used to read data from database
import mysql.connector
import pymysql
from mysql.connector import errorcode
from sqlalchemy import create_engine

# used to store and manipulate data
import pandas as pd
import numpy as np

# files for plotting
import seaborn as sns
import matplotlib.pyplot as plt

# used for plotting data
from plotnine import *
from plydata import *

# used for excluding weekends from plots involving dates
from pandas.tseries.offsets import BDay
isBusinessDay = BDay().is_on_offset

from datetime import datetime

# self created modules for reading sql files and running machine learning algorithms
from sql_db import sql_fun
from ml_analysis import analysis

#for creating word clouds
from collections import Counter
import random
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator

# stop sql warning - pandas is working on fixing it
import warnings
warnings.filterwarnings("ignore")

class visualize():
    ''' This is a class for running machine learning algorithms.
      
    Attributes:
        x_vals (object): The name of the host where the database is located.
        y_vals (object): Username to access the database.'''
        
    def __init__(self, df = None): 
        ''' Initialized analysis object '''
        self.df = df
    
    #plots bar graph of grade 4 and grade 8 maths scores
    def grade_bars(self, scores_df, area1, area2, ave1, ave2, text1, text2, ylim_bottom, ylim_top, legend): 
        
        '''Creates bar graph with lines for national averages

        Parameters:
        scores_df(dataframe): dataframe with scores to graph
        area1(str): name of first subject to plot
        area2(str): name of second subject to plot
        ave1(list): national average score for first area
        ave2(list): national average score for first area
        text1(str): label for line showing first national average score
        text2(str): label for line showing second national average score
        ylim_bottom(int): int for lower limit of y axis
        ylim_top(int): int for upper limit of y axis
        legend(int): int for height of vertical placement of legend
        
        '''
        
        math_df = scores_df[['state', area1, area2, ]]

        fig = plt.figure()
        fig.set_size_inches(8, 5)

        # create bar graph
        ax = math_df.plot.bar(rot=0, x = 'state')
        x = plt.xticks(rotation = 45)

        # zoom in graph to more clearly show score differences
        ax.set_ylim(ylim_bottom, ylim_top)

        # add lines where national averages are
        ax.axhline(ave1, color="red")
        ax.text(1.02, ave1, text1, va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
                transform=ax.get_yaxis_transform())
        ax.axhline(ave2, color="green")
        ax.text(1.02, ave2, text2, va='center', ha="left", bbox=dict(facecolor="w",alpha=0.5),
                transform=ax.get_yaxis_transform())

        # move legend to side of graph
        ax.legend(loc='center left', bbox_to_anchor=(1, legend))

        #show graph
        plt.show()

    def segment_plot(self, state_tests):
        
        '''Creates plot showing gap between online engagement and state assessment score average

        Parameters:
        state_test(dataframe): dataframe with scores to graph
        
        '''

        # calculate an average test score for each state and round it to 2 decimal places
        state_tests['test_ave'] = (state_tests['g4_math_all'] + state_tests['g8_math_all'] + 
                                    state_tests['g4_read_all'] + state_tests['g8_read_all'])/4
        state_tests['average_engagement'] = round(state_tests['engage_average'], 2)

        # calculate the gap between access and engagement to use in the graphic below
        state_tests['gap'] = abs(state_tests['test_ave'] - state_tests['average_engagement'])
        state_tests['label'] = abs(state_tests['test_ave'] + state_tests['average_engagement'])/2

        # Code below is a modified version of the code found on:
        # https://plotnine.readthedocs.io/en/stable/generated/plotnine.geoms.geom_segment.html
        # It creates a graphic illustrating average test scores and engagement by state.

        # creates labels for the plot
        def format_sequence(s, fmt='{}'):
            return [fmt.format(x) for x in s]

        segment_data = (
            state_tests
            >> arrange('-test_ave')
            # Format the floating point data that will be plotted into strings
            >> define(
                test_str='format_sequence(test_ave, "{:.0f}")',
                engage_str='format_sequence(average_engagement, "{:.0f}")',
                gap_str='format_sequence(gap, "{:.0f}")',
            )
        )

        segment_data['test_str'] = 'NAEP: ' + segment_data['test_str']
        segment_data['engage_str'] = 'online: ' + segment_data['engage_str']


        # creates plot of average engagement and test scores
        plot = (ggplot()
         # Range strip
         + geom_segment(
             segment_data,
             aes(x='average_engagement', xend='test_ave', y='state', yend='state'),
             size=10,
             color='#a7a9ac',
             show_legend=False
         )
        #  End point markers
         + geom_point(
             segment_data,
             aes('average_engagement', 'state', color='state', fill='state'),
             size=12,
             stroke=0.7,
             show_legend=False
         )

          + geom_point(
             segment_data,
             aes('test_ave', 'state', color='state', fill='state'),
             size=12,
             stroke=0.7,
             show_legend=False
         )
        #  Add labels to endpoints
         + geom_text(
             segment_data,
             aes(x='test_ave', y='state', label='test_str'),
             size=12,
             ha='center',
             color = 'black'
         )
        + geom_text(
             segment_data,
             aes(x='average_engagement', y='state', label='engage_str'),
             size=12,
             ha='center',
             color = 'black'
        )
         + annotate('text', x=200, y=10.6, label='Average Online Engagement and NAEP Score By State',
                    size=20, color='black', va='top')
         + theme_classic()
         + theme(axis_title_x = element_blank(), axis_title_y = element_blank()) 
         + theme(axis_text_y = element_text(size=20))
         + theme(axis_text_x = element_text(size=20))
         + geom_text(
             segment_data,
             aes(x="label", y='state', label = 'gap_str'),
             size=12,
             fontweight='bold',
             format_string='{}'
         )      
         + theme(figure_size=(16, 10))
        )

        print(plot)
   
    def sort_rules(self, rules):
        
        '''Sorts rules from association rules mining into ones from high scores and low scores

        Parameters:
        rules(dataframe): with rules to sort
     
        Returns
        engage_high(list): list of rows from high values
        high_values(list): list of antecendents from high values
        engage_low(list): list of rows from low values
        low(values): list of antecedents from low values
        '''
        # separates antecedents associated with high and low test data
        engage_high = []
        high_values = []
        engage_low = []
        low_values = []

        for row in rules.itertuples():
            if 'level_high' in row[2]:
                if len(row[2]) == 1:
                    engage_high.append(row[0])
                temp = row[1].split(",")
                for i in range(len(temp)):
                    high_values.append(temp[i].strip())
            if 'level_low' in row[2]:
                if len(row[2]) == 1:
                    engage_low.append(row[0])
                temp = row[1].split(",")
                for i in range(len(temp)):
                    low_values.append(temp[i].strip())
        return engage_high, high_values, engage_low, low_values

    def association_prepare(self, data_frame, level_text, columns):
        
        '''Prepares data for association rules mining

        Parameters:
        dataframe(dataframe): with data to turn into boolean values
     
        Returns
        df(dataframe): dataframe of boolean values
        '''

        # turn engagement average into categorical data
        data_frame['level'] = np.where(data_frame[level_text] >= data_frame[level_text].median(), 'high', 'low')

        data_frame = data_frame.reset_index() # reset index to give a common field to merge on

        dummies = pd.get_dummies(data_frame[['level']]).reset_index() # one hot encode engage levels
        df = pd. merge(data_frame, dummies) #merge dataframes
        df.drop(columns = columns, inplace = True)

        return df

    # random color generator for word cloud
    # function below adapted from: https://stackoverflow.com/questions/43124136/change-the-font-color-of-generated-word-cloud
    
    def random_color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
        '''Changes coloring of words in word cloud. This function was taken from
        https://stackoverflow.com/questions/43124136/change-the-font-color-of-generated-word-cloud
        '''
        num = random.randint(100, 255)  
        return f"hsl({num}, 100%%, %50d%%)" % random.randint(60, 100)

    def make_cloud_list(self, values, title, name="cloud"):
        '''Makes a word cloud from a list

        Parameters:
        values(list): list containing words and values to be turned into a word cloud
        title(str): title for word cloud plot
        name(str): name file to save word cloud to
        '''
    
        # create an analysis object and a word cloud for factors associated with high engagement
        high = analysis()
        cloud = high.word_cloud(lst = values)

        #show the word cloud
        plt.figure(figsize=(40,20))
        plt.imshow(cloud.recolor(color_func=self.random_color_func, random_state=3), interpolation='bilinear')
        plt.title(title, fontsize = 60, color = 'black')
        plt.axis('off')
        plt.savefig(f"{name}.png", dpi=600)
        plt.show()
    
    def make_cloud_dict(self, values, title, name="cloud"):
        '''Makes a word cloud from a dictionary

        Parameters:
        values(dictionary): dictionary containing words and values to be turned into a word cloud
        title(str): title for word cloud plot
        name(str): name file to save word cloud to
        '''

        # create an analysis object and a word cloud for factors associated with high engagement
        high = analysis()
        cloud = high.word_cloud(dictionary = values)

        #show the word cloud
        plt.figure(figsize=(40,20))
        plt.imshow(cloud.recolor(color_func=self.random_color_func, random_state=3), interpolation='bilinear')
        plt.title(title, fontsize = 60, color = 'black')
        plt.axis('off')
        plt.savefig(f"{name}.png", dpi=600)
        plt.show()

    def high_scores(self, data_frame, subject_level, title = None):
        '''Makes a word cloud of products associated with high test scores by engagement index from a data frame

        Parameters:
        data_frame(data_frame): dataframe to be processed and turned into a word cloud
        subject level(str): string for column to base assesment values on
        title(str): title for plot
        '''

        high_df = data_frame.loc[data_frame[subject_level] == 'high']

        high = high_df.groupby(['state', 'name'])['engagement'].mean().reset_index()
        high_sorted = high.sort_values(by = ['state', 'engagement'], ascending = False)

        # use the sorted dataframe to get the ten most used products per district
        top_10_engagement = high_sorted.groupby('state').head(10).reset_index(drop = True)

        # find the engagement average of all products in the top ten by name, create a dictionary for the word cloud
        top_ten_engagement = top_10_engagement.groupby(['name'])['engagement'].mean()
        sums_dict = top_ten_engagement.to_dict()
        
        self.make_cloud_dict(sums_dict, title)

    def low_scores(self, data_frame, subject_level, title = None):
        ''''Makes a word cloud of products associated with low test scores by engagement index from a data frame

        Parameters:
        data_frame(data_frame): dataframe to be processed and turned into a word cloud
        subject level(str): string for column to base assesment values on
        title(str): title for plot
        '''
        high_df = date_frame.loc[data_frame[subject_level] == 'low']

        high = high_df.groupby(['state', 'name'])['engagement'].mean().reset_index()
        high_sorted = high.sort_values(by = ['state', 'engagement'], ascending = False)

        # use the sorted dataframe to get the ten most used products per district
        top_10_engagement = high_sorted.groupby('state').head(10).reset_index(drop = True)

        # find the engagement average of all products in the top ten by name, create a dictionary for the word cloud
        top_ten_engagement = top_10_engagement.groupby(['name'])['engagement'].mean()
        sums_dict = top_ten_engagement.to_dict()
        
        self.make_cloud_dict(sums_dict, title)

    def high_not_low(self, data_frame, variable, score):
        ''''finds products that are used in high performing states, but not low performing ones

        Parameters:
        data_frame(data_frame): dataframe with test scores, product names and engagement indexes
        variable(str): string of columns name to sort by
        score(int): value to separate high and low categories by
        
        Returns
        set of product names only used by high scoring states
        '''
        high_df = data_frame.loc[data_frame[variable] > score]
        low_df = data_frame.loc[data_frame[variable] <= score]

        low_scores = low_df.groupby('name')['engagement'].mean().reset_index()
        low_scores = low_scores.sort_values(by=['engagement'], ascending=False)
        low = list(low_scores['name'].head(20))

        high_scores = high_df.groupby('name')['engagement'].mean().reset_index()
        high_scores = high_scores.sort_values(by=['engagement'], ascending=False)
        high = list(high_scores['name'].head(20))
        
        high_list = list(set(high) - set (low))
        high = " ".join(high_list)

        return high
    
if __name__ == "__main__":
    visualize()