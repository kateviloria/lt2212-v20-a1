# Catherine Viloria - Statistical Methods - Assignment 1


# ADD ANY OTHER IMPORTS YOU LIKE
import argparse
import glob
import numpy as np
import numpy.random as npr
import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# DO NOT CHANGE THE SIGNATURES OF ANY DEFINED FUNCTIONS.
# YOU CAN ADD "HELPER" FUNCTIONS IF YOU LIKE.

def all_files(class_name):
    """
    Creates list of files in directory.

    Args: 
        class_name - directory with class of articles (string)

    Returns:
        A list of all the articles within the directory.
    """

    files_list = []
    for filename in glob.glob('{}/*.txt'.format(class_name)):
        # takes out class name from filename, right now comes out as crude/article001.txt
        filename_split = filename.split('/') 
        filename_no_class = filename_split[1]
        files_list.append(filename_no_class)

    return files_list


def tokenize_dir(class_name):
    """
    Loads all the files in a directory into a list of lists of tokens, one
    list per file. Tokenizes, deletes all punctuation, and lowercases all words.

    Args: 
        class_name - directory with class of articles
    
    Returns:
        A dictionary. The key is the filename (CLASS/ARTICLE*.txt) and its value is a list of strings. 
        Entire article is tokenized (each string is one word from the article.).
        {'CLASSNAME/FILENAME1.txt': ['words', 'more'], 'CLASSNAME/FILENAME2.txt': ['more', 'contents']}
    """

    filename_to_content = {} # dictionary where keys are the filename and value is a list of contents tokenized

    for filename in glob.glob('{}/*.txt'.format(class_name)):
        words = [] 
        only_letters = [] 
        with open(filename, "r") as textfile:
            for line in textfile.readlines():
                words = line.split(' ')
                for every_word in words:
                    if every_word.isalpha(): # filters out integers and punctuation
                        no_capitals = every_word.lower() # makes each word into lowercase
                        only_letters.append(no_capitals)
        filename_to_content[filename] = only_letters # assigns filename as key and its value the tokenized contents of each article
    
    return filename_to_content


def load_both_dir(class_name, class_name2):
    """
    Amalgamates the dictionaries made for the two classes (directories) provided.

    Args:
        class_name - first directory with class of articles 
        class_name2 - second directory with different class of articles

    Returns:
        A dictionary that contains both dictionaries made per class. Each key is a string that 
        represents the filename and its value is a list of strings that are all the frame IDs within 
        the article.
        {'CLASSNAME_FILENAME.txt' : ['word', 'another'], 'ANOTHERCLASSNAME_FILENAME.txt' : ['more', 'words']}
    """

    class_one_tokens = tokenize_dir(class_name)
    class_two_tokens = tokenize_dir(class_name2)

    both_classes_dict = class_one_tokens

    both_classes_dict.update(class_two_tokens) # amalgamates both classes into one dictionary

    return both_classes_dict


def count_freq(class_name, class_name2):
    """
    Counts how many times each word appears in each article. 

    Args:
        class_name - first directory with class of articles
        class_name2 - second directory with different class of articles

    Returns:
        A sorted list with all the unique words that appear in the corpus.

        AND

        A list of dictionaries where each dictionary represents one article. Two of the keys in 
        each dictionary are 'class_of_article' (directory) and 'article_name' (filename)
        and the values are the corresponding information respective of the article. 
        The rest of the keys are words that appear in the article and the values are their 
        frequencies (integers). Also returns a list with all the words that appear in all articles.
        [{'class_of_article': 'crude', 'article_name': 'article001.txt', 'a_word': 5, 'another_word': 8}]
    """

    class_list = [class_name, class_name2]
    all_classes = load_both_dir(class_name, class_name2)
    all_possible_words = [] # list with all words in all articles
    all_counts = []

    for items in all_classes.items():
        file_info = items[0]
        file_info_list = file_info.split('/')
        name_of_class = file_info_list[0]
        name_of_file = file_info_list[1]

        words = items[1] # value of dictionary (list of words in article)

        article_dict = {}

        # adds key and value of file info (both class and article)
        article_dict['class_of_article'] = name_of_class 
        article_dict['article_name'] = name_of_file

        for every_word in words:
            all_possible_words.append(every_word)
            if every_word in article_dict.keys():
                article_dict[every_word] += 1
            else:
                article_dict[every_word] = 1
        all_counts.append(article_dict)
        unique_words = list(set(all_possible_words)) # takes out all repeated words and makes a list
        sorted_unique = sorted(unique_words) #alphabetizes words
        
    return sorted_unique, all_counts


def make_pandas(class_name, class_name2):
    """
    Makes list of dictionaries into a pandas table. 

    Args:
        class_name - first directory with class of articles
        class_name2 - second directory with different class of articles

    Returns:
        A pandas table where the columns are class, filename, and all words that have appeared
        in the corpus. The rows are the articles and the following frequencies for each word
        within each article. 
    """

    sorted_unique = count_freq(class_name, class_name2)[0] # list of all words in articles
    all_counts = count_freq(class_name, class_name2)[1]

    final_table = pd.DataFrame(all_counts)

    # change all frequencies from floats into integers, change NaN to 0
    final_table[sorted_unique] = final_table[sorted_unique].fillna(0.0).astype(int)

    return final_table


def part1_load(folder1, folder2, n=1):
    """
    Makes a table with only words that show up n amount of times in the corpus and its counts
    within each article.

    Args:
        class_name - first directory with class of articles
        class_name2 - second directory with different class of articles
        n - integer for how many times word should appear in entire corpus (has a default of 1)

    Returns:
        A pandas table where the columns are class, filename, and all the words that have appeared
        in the corpus. The rows are the article names and the following frequencies for each word
        within each article. 
    """

    pandas_table = make_pandas(folder1, folder2)
    sorted_unique = count_freq(folder1, folder2)[0] # list of all words in articles

    words_to_drop = [] # list of words that are not equal to or less than n

    for column in pandas_table[sorted_unique]:
        times_appeared_in_corpus = pandas_table[column].sum() # totals values in each column
        if times_appeared_in_corpus <= n: 
            words_to_drop.append(column)

    new_pandas_table = pandas_table.drop(words_to_drop, axis=1)
        
    return new_pandas_table 


def part2_vis(df, m=1):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    # CHANGE WHAT YOU WANT HERE
    """
    Creates a bar chart that uses top m term frequencies in DataFrame with matching bars per class.

    Args: 
        df - pandas dataframe
        m - top m term frequencies
    
    Returns:
        A bar graph. X axis is each word (each bar is the class). Y axis is frequency.
    """

    # add part one load to check as df
    pandas_table = df

    # drop 'article_name'
    pandas_table = pandas_table.drop('article_name', axis=1)

    #class frequencies 
    group_table = pandas_table.groupby(['class_of_article']).sum() # groups by class
    
    transposed_table = group_table.T # switches rows and columns (rows = words, columns = classes)
   
    column_names = [] # classes 
    for column in transposed_table.columns:
        column_names.append(column)

    transposed_table['sums'] = transposed_table[column_names].sum(axis=1) # add sums column

    # sorts in descending order and takes top m frequencies
    ordered_table = transposed_table.sort_values(by=['sums'], ascending=False).head(m) # sums frequency of word in both classes
    final_table = ordered_table.drop('sums', axis=1)

    bar_graph = final_table.plot.bar()

    return bar_graph 


def calc_idf(df):
    """
    Calculates IDF.

    Args:
        df - pandas table

    Returns:
        A numpy array of IDF for each word. 
    """

     # add part one load to check as df
    pandas_table = df

    total_docs = pandas_table.shape[0] # counts number of files through number of rows

    pandas_table = np.split(pandas_table, [2], axis=1) # splits table into two
    file_info_table = pandas_table[0]
    word_table = pandas_table[1]

    # total number of articles each word (column) appears in
    docs_word_is_in = word_table[word_table > 0].count() 

    # total number of docs divided by total number of articles each word appears in
    dividing_docs = np.divide(total_docs, docs_word_is_in)

    # multiply by natural log
    idf = np.log(dividing_docs)

    return idf


def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    # CHANGE WHAT YOU WANT HERE
    """
    Calculates tf-idf.

    Args:
        df - pandas table (columns: class_of_article, article_name, all words in article; rows: frequencies)

    Returns:
        Pandas table with tf-idf for each word within each article.
    """

    # add part one load to check as df
    pandas_table = df
    pandas_table_operate = df

    idf = calc_idf(pandas_table_operate) # idf first so pandas_table doesn't change

    # using raw count, separate file info and raw counts
    pandas_table = np.split(pandas_table, [2], axis=1)
    file_info_table = pandas_table[0]
    word_table = pandas_table[1]


    tf_idf = word_table.multiply(idf, axis=1) # multiplies tf and idf

    tfidf_result = pd.concat([file_info_table, tf_idf], axis=1,) # merges info table and tf-idf table

    return tfidf_result


# ADD WHATEVER YOU NEED HERE, INCLUDING BONUS CODE.

def part_bonus(df):
    """
    Runs a classifier and evaluates the classification accuracy on the training data.

    Args:
        df - pandas table 
    
    Returns:
        An evaluation score. 
    """

    dataset = df

    # preprocessing 
    data = dataset.drop(['class_of_article', 'article_name'], axis=1)
    target = dataset['class_of_article']
    
    # creates training sets and test sets
    data_train, data_test, target_train, target_test = train_test_split(data, target, train_size = 0.80, random_state=10)

    svc_model = SVC(random_state=100)

    # train using training data and predict using test data
    pred = svc_model.fit(data_train, target_train).predict(data_test)

    accuracy = accuracy_score(target_test, pred, normalize=True)

    print('Accuracy: ', accuracy)