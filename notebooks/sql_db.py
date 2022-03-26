#!/usr/bin/env python
# coding: utf-8

'''Interact with or create a mySQL database. Connect, create tables, add data to tables and query.

Classes:

    sql_fun()

Functions:

    connect () -> object
    create_tables(string, string, string)
    df_to_sql(string, object)
    query(string) -> object
'''

# In[ ]:
import mysql.connector
import pymysql
from mysql.connector import errorcode
from sqlalchemy import create_engine

# used to store and manipulate data
import pandas as pd
import numpy as np

class sql_fun():

    """
    This is a class for interacting with a mySQL database.
      
    Attributes:
        host (str): The name of the host where the database is located.
        user (str): Username to access the database.
        password (str): The password for the database
        database (str): The name of the database to be accesses

    """
    
    def __init__(self, host, user, password, database):

        """

        Initializes sqlfun object

        Parameters:
            host (str): The name of the host where the database is located.
            user (str): Username to access the database.
            password (str): The password for the database
            database (str): The name of the database to be accesses
        """

        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect(self):
        '''creates a connection to the database'''
        conn = None
        try:
            conn = mysql.connector.connect(
                host = self.host,
                user = self.user,
                passwd = self.password,
                database = self.database
                )
            
        except:
            
            conn = mysql.connector.connect(
            host = self.host,
            user = self.user,
            passwd = self.password,
            )
            
            c = conn.cursor()
            c.execute("CREATE DATABASE " +  self.database + ";")
            conn.close()
                
            conn = mysql.connector.connect(  
            host = self.host,
            user = self.user,
            passwd = self.password,
            database = self.database
            )
            print("Creating new database.")
            
        return conn


    def create_tables(self, scripts, beg_conditions = None, final_conditions = None):
        '''creates new tables in a mysql database.

        Parameters:
        scripts(list): list of strings containing the SQL script for 
        each table to be created
        beg_conditions(list): list of strings containing the SQL script for
        any conditions to be set before tables are created
        final_conditions(list): list of strings containing the SQL script for
        any conditions to be set before tables are created '''
        
        i = 0 #used to track number of tables created
        completed = False
        conn = self.connect()
        c = conn.cursor() 
        try:
            if beg_conditions != None:
                for condition in beg_conditions:
                    c.execute(condition)
                conn.commit()
            for script in scripts:
                c.execute(script)
                conn.commit()
                print(f'{i} table created.')
                i += 1
            if final_conditions !=  None:
                for condition in final_conditions:
                    c.execute(condition)
                conn.commit()
            conn.close()
            completed = True
        except:
            print(f"There was an error creating tables. Only {i} tables were created.")
                
        return completed
    
    def df_to_sql(self, table, dataframe):
        '''adds data from dataframe to given table in mysql.

        Parameters:
        table(str): name of the table where data should be added
        dataframe(dataframe): dataframe containing data to add to sql
            all column names in dataframe must match column names in table exactly'''
        
        conn = self.connect()
        #creates and engine that is supposed to make upload faster
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(
                                   user = self.user,
                                   pw = self.password,
                                   db = self.database,
                                   fast_execute_many = True)) 
        #adds dataframe data to database
        dataframe.to_sql(name = table, con = engine, if_exists = "append", index = False)
        print(f"Records have been inserted into the {table} table.")
        conn.commit()
        conn.close()
        
    def dflist_to_sql(self, table, dflist):
        '''adds data from dataframe to given table in mysql.

        Parameters:
        table(str): name of the table where data should be added
        dataframe(list): list of dataframes containing data to add to sql
            all column names in dataframes must match column names in table exactly'''
        
        conn = self.connect()

        #creates and engine that is supposed to make upload faster
        engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(
                                   user = self.user,
                                   pw = self.password,
                                   db = self.database,
                                   fast_execute_many = True))
        
        #adds data from each frame in list file to table
        i = 0
        for df in dflist:
            df.to_sql(name = table, con = engine, if_exists = "append", index = False)
        print(f'{i} records have been inserted into the {table} table.')
        conn.commit()
        conn.close()
        
    def query(self, script):
        '''queries mysql database.

        Parameters:
        script(str): string of queryscript to use
        
        Returns:
        results of query in the form of a pandas dataframe'''
        
        conn = self.connect()
        result = pd.read_sql(script, conn)

        return result
        
            

