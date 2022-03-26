#!/usr/bin/env python
# coding: utf-8

'''Run often used Machine Learning Algorithms.

Classes:

    analysis()

Functions:

    linear_regression
    decision_tree()
    decision_tree_balanced(int)
'''

# used to store and manipulate data
import pandas as pd
import numpy as np

# used to plot data
import seaborn as sns
import matplotlib.pyplot as plt

#https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

#used to create training and testing sets
from sklearn.model_selection import train_test_split 

#for creating word clouds
import random
from collections import Counter
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator


#used to create decision tree models
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 

# used to determine accuracy of models
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# files needed for linear regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

# for association rules mining
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

class analysis():
    ''' This is a class for running machine learning algorithms.
      
    Attributes:
        x_vals (object): The name of the host where the database is located.
        y_vals (object): Username to access the database.'''
    
    def __init__(self, x_vals = None, y_vals = None): 
        ''' Initialized analysis object

        Attributes:
            x_vals (object): The dataframe columns that will be used for test and training values.
            y_vals (object): The dataframe column that will be used for test and training values.'''

        self.x_values = x_vals
        self.y_values = y_vals
        
    def linear_regression(self):
        '''runs linear regression algorithm on x and y values.'''
        
        #Look at the correlation between scores and engagement
        X = self.x_values
        y = self.y_values.values.reshape(-1,1)

        #divide data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        #create the model and run it with the training sets
        model = LinearRegression()
        model.fit(X_train, y_train)

        # evaluate the model
        y_pred = model.predict(X_test)

        # evaluate predictions
        mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
        squared_error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        coefficient = model.score(X, y)

        #print the results
        print('Mean Absolute Error: %.3f' % mean_absolute_error)
        print('Squared Error: %.3f' % squared_error)
        print('Coefficient of determination: %.3f' % coefficient)

        #plot the results
        for col in X.columns:
            plt.scatter(X[col], y, s=10)
        
        
    def decision_tree(self, depth):
        '''runs decision tree algorithm on x and y values.

        Parameters:
        depth(int): depth of decision tree'''
         
        #select features to compare 
        X = self.x_values
        y = self.y_values

        #divide data into testing and training sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(max_depth= depth)

        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        #print confusion matrix
        unique_label = np.unique([y_test, y_pred])
        print()
        print("Confusion Matrix")
        print()
        cmtx = pd.DataFrame(
            metrics.confusion_matrix(y_test, y_pred, labels=unique_label), 
            index=['true:{:}'.format(x) for x in unique_label], 
            columns=['pred:{:}'.format(x) for x in unique_label])
        print(cmtx)
        
        
        
    def decision_tree_balanced(self, depth, over_vals, over_nums, under_vals, under_nums):
        '''used for creating a decision tree. Balances unbalanded data first.

        Parameters:
        depth(num): depth of decision tree
        over_vals(list): list containing names of columns to be oversampled
        over_nums(list): list containing number of samples for each value in over_vals
        under_vals(list): list containing names of columns to be undersampled
        under_nums(list): list containing number of samples for each value in under_vals'''
        
        
        #select features 
        X = self.x_values
        y = self.y_values

        #divide data into testing and training sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        #over and undersample to even-out imbalanced dataset, pipeline code taken from here: 
        #https://machinelearningmastery.com/combine-oversampling-and-undersampling-for-imbalanced-classification/
        over_strategy = dict(zip(over_vals, over_nums))
        under_strategy = dict(zip(under_vals, under_nums))
        
        over = SMOTE(sampling_strategy = over_strategy)
        under = RandomUnderSampler(sampling_strategy = under_strategy)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)

        # transform the dataset
        X_train, y_train = pipeline.fit_resample(X_train, y_train)

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier(max_depth= 7)

        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        #print confusion matrix
        unique_label = np.unique([y_test, y_pred])
        print()
        print("Confusion Matrix")
        print()
        cmtx = pd.DataFrame(
            metrics.confusion_matrix(y_test, y_pred, labels=unique_label), 
            index=['true:{:}'.format(x) for x in unique_label], 
            columns=['pred:{:}'.format(x) for x in unique_label])
        print(cmtx)
        
        
    def association_apriori(self, data_frame, support = 0.5, length = 3, metric = 'lift', threshold = 2):
        
        '''used for association rules mining with the Apriori algorithm.

        Parameters:
        data_frame(object): data frame that has been one-hot encoded
        support(float): list containing names of columns to be oversampled
        length(int): list containing number of samples for each value in over_vals
        metric(str): list containing names of columns to be undersampled
        threshold(float): list containing number of samples for each value in under_vals'''

        # use the apriori algorithm to mine association rules. Lower support finds more rules.
        associations = apriori(data_frame, min_support = support, use_colnames=True, max_len=length)

        # get a database of the rules that meet the mininmum threshold you select.
        apriori_df = association_rules(associations, metric=metric, min_threshold=threshold,  support_only=False)
        
        # change frozen sets into strings
        apriori_df["antecedents"] = apriori_df["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode") 
        apriori_df["consequents"] = apriori_df["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode") 
        
        return apriori_df 
    
    def association_fpgrowth(self, data_frame, support = 0.5, metric = 'lift', threshold = 2):
        
        '''used for association rules mining with the fpGrowth algorithm.

        Parameters:
        data_frame(object): data frame that has been one-hot encoded
        support(float): list containing names of columns to be oversampled
        metric(str): list containing names of columns to be undersampled
        threshold(float): list containing number of samples for each value in under_vals'''

        # use the apriori algorithm to mine association rules. Lower support finds more rules.
        associations = fpgrowth(data_frame, min_support = support, use_colnames=True)

        # get a database of the rules that meet the mininmum threshold you select.
        growth_df = association_rules(associations, metric=metric, min_threshold=threshold,  support_only=False)
        
        # change frozen sets into strings
        growth_df["antecedents"] = growth_df["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode") 
        growth_df["consequents"] = growth_df["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode") 

        return growth_df
    
    def word_cloud(self, dictionary = None, lst= None, image = None, background = 'black', contour = 'pink', name = "cloud"):
        '''used for creating word clouds.

        Parameters:
        lst(object): list object
        image(object): image to use as a mask to shape word cloud
        background(str): string for background color
        contour(str): string for contour color
        '''
        
        # assign either the list or dictionary to words variable for the word cloud
        words = ['you', 'forgot' 'the' 'word' 'list']
        
        if list != None:
            words = Counter(lst)
            
        if dictionary != None:
            words = dictionary
            
        
        # create the WordCloud object
        wordcloud = WordCloud(min_word_length =3,
                                  background_color=background,
                                  random_state = 42,
                                  contour_width = 1, 
                                  width=800, 
                                  height=400,
                                  contour_color = contour)
        
        # create new WordCloud object with mask if one provided
        if image != None:
            mask = np.array(Image.open(image))
            wordcloud = WordCloud(min_word_length =3,
                                  mask=mask,
                                  background_color=background,
                                  width=mask.shape[1],
                                  height=mask.shape[0],
                                  random_state = 42,
                                  contour_width = 1, 
                                  contour_color = contour)
                                      
        # generate the word cloud
        
        cloud =  wordcloud.generate_from_frequencies(words)
            
        return cloud
    

if __name__ == "__main__":
    analysis()
