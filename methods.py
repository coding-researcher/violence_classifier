import sys
import csv
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pickle
import re
import numpy as np
import math
import matplotlib.pyplot as plt
import collections


def read_tweet_urls(csvfile, tweetlist):
    '''
	Given the name of a CSV file containing tweets and their verbal violence scale codes,
	open the file, read the tweets, and return the tweet IDs and ALL item codes in a list. 
    '''
    with open(csvfile, 'r') as coded_tweets:
        reader = csv.DictReader(coded_tweets)
        for row in reader:
            code = [row['Input.url'], row['Answer.entry.1'], row['Answer.entry.2'], row['Answer.entry.3'], row['Answer.entry.4'], row['Answer.entry.5'], row['Answer.entry.6'], row['Answer.entry.7'], row['Answer.entry.8'], row['Answer.entry.9'], row['Answer.entry.10'], row['Answer.entry.11'], row['Answer.entry.12'], row['Answer.entry.13'], row['Answer.entry.14']]
            tweetlist.append(code)

def read_tweets(csvfile, tweetlist, entryfield):
    '''
    Given the name of a CSV file and a verbal violence scale item, return a list of tweet IDs
    with the classifications for the scale item specified.
	'''
    with open(csvfile, 'r') as coded_tweets:
        reader = csv.DictReader(coded_tweets)
        for row in reader:
            code = (row['Input.tweet'], row[entryfield])
            tweetlist.append(code)
            
def sample_tweets(tweetlist, testingindices):
    traininglist = []
    testinglist = []
    for tweetnum in range(len(tweetlist)):
        if tweetnum in testingindices:
            traininglist.append(tweetlist[tweetnum])
        else:
            testinglist.append(tweetlist[tweetnum])
    return traininglist, testinglist

# pull tweet objects matching ids in a list of tweet ids
def get_full_tweets_from_ids(cursor, ids, start, end):
    tweetlist = []
    id_list = ids[start:end]
    print('start = %s, end = %s' % (start, end))
    print('cursor warnings before id_list loop: %s ' % str(cursor.fetchwarnings()))
    for item in id_list:
        sql = 'SELECT * FROM `tweet` WHERE `tweet_id_str` LIKE ' + item + ' LIMIT 1'
        print('cursor warnings before execute: %s ' % str(cursor.fetchwarnings()))
        cursor.execute(sql)
        print('cursor warnings after execute: %s ' % str(cursor.fetchwarnings()))
        result = cursor.fetchall()
        #result = cursor.fetchone()
        print('cursor warnings after fetch: %s ' % str(cursor.fetchwarnings()))
        tweetlist.append(result)
    print('cursor warnings before return: %s ' % str(cursor.fetchwarnings()))
    return tweetlist

def get_label_array(tweet_labels):
    """Return a *numpy array* of ints for the coded labels of each tweet.
    1 means uncharacteristic, 2 means not sure, 3 means characteristic. 
    Params:
        tweet_labels....a list of labels for the batch of tweets
    Returns:
        0 = uncharacteristic/not sure and 1 = characteristic
        #a numpy array of 1, 2, or 3 values corresponding to each element
        #of tweet labels, where 1 indicates a positive review, and 0
        #indicates a negative review.
    """
    labels = np.zeros(len(tweet_labels), np.int_)
    for x in range(len(tweet_labels)):
        #print(tweet_labels[x])
        if tweet_labels[x] == '3':
            #print("characteristic")
            labels[x] = 1
    return labels


def tokenize(text):
    """Given a string, return a list of tokens such that: (1) all
    tokens are lowercase, (2) all punctuation is removed. Note that
    underscore (_) is not considered punctuation.
    Params:
        text....a string
    Returns:
        a list of tokens
    """
    lower = text.lower()
    no_urls = re.sub('http\S+', 'THIS_IS_A_URL', lower)
    tokens = re.findall('\w+', no_urls)
    return tokens
    
def tokenize_with_punct(text):
    """Given a string, return a list of tokens such that: (1) all
    tokens are lowercase, (2) all punctuation is kept as separate tokens.
    Note that underscore (_) is not considered punctuation.
    Params:
        text....a string
    Returns:
        a list of tokens
    """
    lower = text.lower()
    tokens = re.findall('\w+|[^\w\s\n\r\x85]', lower)
    return tokens

def do_vectorize(filenames, tokenizer_fn=tokenize, min_df=1,
                 max_df=1., binary=True, ngram_range=(1,1)):
    """
    Convert a list of filenames into a sparse csr_matrix, where
    each row is a file and each column represents a unique word.
    Use sklearn's CountVectorizer: http://goo.gl/eJ2PJ5
    Params:
        filenames.......list of review file names
        tokenizer_fn....the function used to tokenize each document
        min_df..........remove terms from the vocabulary that don't appear
                        in at least this many documents
        max_df..........remove terms from the vocabulary that appear in more
                        than this fraction of documents
        binary..........If true, each documents is represented by a binary
                        vector, where 1 means a term occurs at least once in 
                        the document. If false, the term frequency is used instead.
        ngram_range.....A tuple (n,m) means to use phrases of length n to m inclusive.
                        E.g., (1,2) means consider unigrams and bigrams.
    Return:
        A tuple (X, vec), where X is the csr_matrix of feature vectors,
        and vec is the CountVectorizer object.
    """
    vectorizer = CountVectorizer(tokenizer=tokenizer_fn, min_df=min_df, max_df=max_df, binary=binary, ngram_range=ngram_range, dtype=int)
    X = vectorizer.fit_transform(filenames)
    return (X, vectorizer)

def vectorize_2(filenames, tokenizer_fn=tokenize, min_df=1,
                 max_df=1., binary=True, ngram_range=(1,1)):
    """
    Convert a list of filenames into a sparse csr_matrix, where
    each row is a file and each column represents a unique word.
    Use sklearn's CountVectorizer: http://goo.gl/eJ2PJ5
    """
    vectorizer = CountVectorizer(tokenizer=tokenizer_with_punct, min_df=1, max_df=1., binary=True, ngram_range=(1,1), dtype=int)
    X = vectorizer.fit_transform(filenames)
    return (X, vectorizer)

def get_balanced_clf():
    return LogisticRegression(random_state=42, class_weight='balanced')

def get_clf():
    return LogisticRegression(random_state=42)

def accuracy(truth, predicted):
    return (1. * len([1 for tr, pr in zip(truth, predicted) if tr == pr]) / len(truth))

def do_cross_validation(X, y, n_folds=10, verbose=False):
    """
    Perform n-fold cross validation, calling get_clf() to train n
    different classifiers. Use sklearn's KFold class: http://goo.gl/wmyFhi
    Be sure not to shuffle the data, otherwise your output will differ.
    Params:
        X.........a csr_matrix of feature vectors
        y.........the true labels of each document
        n_folds...the number of folds of cross-validation to do
        verbose...If true, report the testing accuracy for each fold.
    Return:
        the average testing accuracy across all folds.
    """
    # changed to stratified k-fold to have the same # of positive examples in each fold.
    #kf = StratifiedKFold(y.size, n_folds)
    kf = KFold(y.size, n_folds)
    #model = get_clf()
    model = get_balanced_clf()
    accuracies = []
    try:
        for train_ind, test_ind in kf:
            model.fit(X[train_ind], y[train_ind])
            predictions = model.predict(X[test_ind])
            #accuracies.append(roc_auc_score(y[test_ind], predictions)
            accuracies.append(roc_auc_score(y[test_ind], predictions, average='weighted'))
    except:
        sys.exit('not enough positive examples')
            

    if verbose:
        for n in range(0, n_folds):
            print ("fold", n, "accuracy=", accuracies[n])
        
    return np.mean(accuracies)

def do_expt(filenames, y, tokenizer_fn=tokenize,
            min_df=1, max_df=1., binary=True,
            ngram_range=(1,1), n_folds=10):
    """
    Run one experiment, which consists of vectorizing each file,
    performing cross-validation, and returning the average accuracy.
    You should call do_vectorize and do_cross_validation here.
    Params:
        filenames.......list of review file names
        y...............the true sentiment labels for each file
        tokenizer_fn....the function used to tokenize each document
        min_df..........remove terms from the vocabulary that don't appear
                        in at least this many documents
        max_df..........remove terms from the vocabulary that appear in more
                        than this fraction of documents
        binary..........If true, each documents is represented by a binary
                        vector, where 1 means a term occurs at least once in 
                        the document. If false, the term frequency is used instead.
        ngram_range.....A tuple (n,m) means to use phrases of length n to m inclusive.
                        E.g., (1,2) means consider unigrams and bigrams.
        n_folds.........The number of cross-validation folds to use.
    Returns:
        the average cross validation testing accuracy.
    """
    matrix, vec = do_vectorize(filenames, tokenizer_fn, min_df, max_df, binary, ngram_range)
    accuracy = do_cross_validation(matrix, y, n_folds)
    return accuracy

def model_train_setup(scale_item, labels, indices, balanced=True):
    coding_train, coding_test = sample_tweets(labels, indices)
    
    training_text = [x[0] for x in coding_train]
    training_labels = [x[1] for x in coding_train]
    
    if scale_item == 2:
        X, vec = vectorize_2(training_text)
    else:
        X, vec = do_vectorize(training_text)
    y = get_label_array(training_labels)
    
    if(balanced):
        clf = get_balanced_clf()
    else:
        clf = get_clf()
        
    clf.fit(X, y)
    
    print('average cross validation accuracy=%.4f\n' % do_cross_validation(X, y))
    return coding_train, coding_test, X, y, vec, clf

def train_and_tweak_model(labeled_data, indices_of_labeled_data, scale_item):
    try:
        training_data, testing_data, X, y, vec, clf = model_train_setup(scale_item, labeled_data, indices_of_labeled_data)
        training_text = [x[0] for x in training_data]
        model_tweaks(training_text, y)
    except Exception as e:
        print('no model available: %s' % str(e))

def model_tweaks(training_text, y):
    print('1', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=1, max_df=1., binary=True, ngram_range=(1,1), n_folds=10))
    print('2', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=1, max_df=1., binary=True, ngram_range=(1,1), n_folds=10))
    print('3', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=1, max_df=1., binary=True, ngram_range=(3,3), n_folds=10))
    print('4', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=1, max_df=1., binary=True, ngram_range=(2,2), n_folds=10))
    print('5', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=2, max_df=1., binary=True, ngram_range=(3,3), n_folds=10))
    print('6', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=1, max_df=.9, binary=True, ngram_range=(2,2), n_folds=10))
    print('7', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=2, max_df=.9, binary=False, ngram_range=(3,3), n_folds=10))
    print('8', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=1, max_df=.9, binary=False, ngram_range=(3,3), n_folds=10))
    print('9', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=2, max_df=1., binary=True, ngram_range=(2,2), n_folds=10))
    print('10', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=2, max_df=.9, binary=False, ngram_range=(2,2), n_folds=10))
    print('11', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=2, max_df=.9, binary=False, ngram_range=(2,3), n_folds=10))
    print('12', do_expt(training_text, y, tokenizer_fn=tokenize, min_df=1, max_df=1., binary=False, ngram_range=(2,3), n_folds=10))
    print('13', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=1, max_df=1., binary=True, ngram_range=(3,3), n_folds=10))
    print('14', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=1, max_df=1., binary=True, ngram_range=(2,2), n_folds=10))
    print('15', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=2, max_df=1., binary=True, ngram_range=(3,3), n_folds=10))
    print('16', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=1, max_df=.9, binary=True, ngram_range=(2,2), n_folds=10))
    print('17', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=2, max_df=.9, binary=False, ngram_range=(3,3), n_folds=10))
    print('18', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=1, max_df=.9, binary=False, ngram_range=(3,3), n_folds=10))
    print('19', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=2, max_df=1., binary=True, ngram_range=(2,2), n_folds=10))
    print('20', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=2, max_df=.9, binary=False, ngram_range=(2,2), n_folds=10))
    print('21', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=2, max_df=.9, binary=False, ngram_range=(2,3), n_folds=10))
    print('22', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=1, max_df=1., binary=False, ngram_range=(2,3), n_folds=10))
    print('23', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=1, max_df=1., binary=True, ngram_range=(2,3), n_folds=10))
    print('24', do_expt(training_text, y, tokenizer_fn=tokenize_with_punct, min_df=1, max_df=1., binary=True, ngram_range=(1,3), n_folds=10))

def model_test_expt(test_data, vectorizer, clf):
    testing_labels = [x[1] for x in test_data]
    testing_text = [x[0] for x in test_data]

    y_test = get_label_array(testing_labels)
    X_test = vectorizer.transform(testing_text)
    y_pred = clf.predict(X_test)

    try:
        print('testing accuracy = %.4g' % accuracy_score(y_test, y_pred))
        print('f1 score = %.4g' % f1_score(y_test, y_pred))
        print('roc auc score = %.4g' % roc_auc_score(y_test, y_pred))
    except Exception:
        print('not enough positive examples')
        return 0, 0, 0

    return X_test, y_test, y_pred


def train_and_test_model(labeled_data, indices_of_labeled_data, scale_item):
    try:
        training_data, testing_data, X, y, vec, clf = model_train_setup(scale_item, labeled_data, indices_of_labeled_data)
        #print('train setup complete')
        X_test, y_true, y_pred = model_test_expt(testing_data, vec, clf)
        evaluate_model(X, y, X_test, y_pred, clf, vec, training_data, testing_data)
        print('test expt complete')
    except Exception as e:
        print('no model available: %s' % str(e))


def get_terms(vocab):
    return np.array([x[0] for x in sorted(vocab.items(), key=lambda x: x[1])])

def get_top_coefficients(clf, vec, n=10):
    """ Get the top n coefficients for each class (positive/negative).
    Params:
        clf...a LogisticRegression object that has already been fit to data.
        vec...a CountVectorizer
        n.....the number of features to print per class.
    Returns:
        Two lists of tuples. The first list containts the top terms for the positive
        class. Each entry is a tuple of (string, float) pairs, where
        string is the feature name and float is the coefficient.
        The second list is the same but for the negative class.
        In each list, entries should be sorted in descending order of 
        absolute value."""
    terms = get_terms(vec.vocabulary_)
    coef = clf.coef_[0]
    #coef_char = clf.coef_[1]
    srted = np.argsort(coef)
    #sorted_char = np.argsort(coef_car)
    top_char = srted[::-1][:n]
    top_un = srted[:n]
    un = (terms[top_un], coef[top_un])
    char = (terms[top_char], coef[top_char])
    return un, char

def get_top_errors(X_test, y_test, clf, filenames, n=10):
    """
    Use clf to predict the labels of the testing data in X_test. 
    We want to find incorrectly predicted documents. Furthermore, we want to look at those 
    where the probability of the incorrect label, according to the classifier, is highest.
    Use the .predict_proba method of the classifier to get the probabilities of
    each class label. Return the n documents that were misclassified, sorted by the
    probability of the incorrect label. The returned value is a list of dicts, defined below.
    Params:
        X_test......the testing matrix
        y_test......the true labels for each testing document
        filenames...the filenames for each testing document
        clf.........a trained LogisticRegression object
        n...........the number of errors to return
    Returns:
        A list of n dicts containing the following key/value pairs:
           index: the index of this document (in the filenames array)
           probas: a numpy array containing the probability of class 0 and 1
           truth: the true label
           predicted: the predicted label
           filename: the path to the file for this document
    """
    dicts = []
    wrong = []
    predicted = clf.predict(X_test)
    probas = clf.predict_proba(X_test)
    # find all features where predicted label does not match true label
    # add all output attributes to a list of wrong features
    for i in range(len(predicted)):
        for j in range(0,2):
            #probability = probas[i][j]
            if predicted[i] != y_test[i]:
                wrong.append((i, probas[i], y_test[i], predicted[i], filenames[i]))
    
    # sort list of wrong features by highest probability that predicted label is not true label
    most_wrong = sorted(wrong, key=lambda x: x[4], reverse=True)

    # create dictionary object for top n errors and add to output list of dictionaries
    for j in range(n):
        obj = {}
        obj['index'] = most_wrong[j][0]
        obj['probas'] = most_wrong[j][1]
        obj['truth'] = most_wrong[j][2]
        obj['predicted'] = most_wrong[j][3]
        obj['text'] = most_wrong[j][4]
        dicts.append(obj)   
    return dicts

# this is definitely not accurate for 3 classes. How to fix?
def most_predictive_term_in_doc(instance, clf, class_idx):
    """
    Params:
        instance....one row in the X csr_matrix, corresponding to a document.
        clf.........a trained LogisticRegression classifier
        class_idx...0 or 1. The class for which we should find the most 
                    predictive term in this document.
    Returns:
        The index corresponding to the term that appears in this instance
        and has the highest coefficient for class class_idx.
    """
    terms = instance.indices
    coefficients = clf.coef_[0]
    
    instance_coefficients = []
    for t in terms:
        instance_coefficients.append((t, coefficients[t]))
    
    # sort so highest coefficient (pos or neg) is first
    if (class_idx == 0):
        srted = sorted(instance_coefficients, key=lambda x: x[1])
    else:
        srted = sorted(instance_coefficients, key=lambda x: x[1], reverse=True)
    
    highest = srted[0]
    
    return int(highest[0])

def find_contexts(filename, term, window=5):
    """
    Find all context windows in which this term appears in this file.
    You should use tokenize_with_not to tokenize this file. 
    
    Params:
        filename....the filename for this document.
        term........the term to find
        window......return this many tokens to the left and this many tokens to
                    the right of every occurrence of term in this document
    Returns:
        a list of strings. Each string contains the matched context window.
    """
    string_list = []
    tokens = tokenize(filename)
    string = ""
    
    for x in range (len(tokens)):
        if tokens[x] == term:
            seq = []
            for y in range(x-window, x+window+1):
                try:
                    tok = tokens[y]
                    seq.append(tok)
                except IndexError:
                    tok = ""           
            string = ' '.join(seq)
            string_list.append(string)
    return string_list


def print_errors(errors, clf, X, vec, window=5):
    for error in errors:
        fidx = most_predictive_term_in_doc(X[error['index']], clf, error['predicted'])
        term = vec.get_feature_names()[fidx]
        print('document %s misclassified as %d' % (error['text'], error['predicted']))
        print('%s appears here:' % (term))
        print(find_contexts(error['text'], term, window))
        print('')


def evaluate_model(X, y, X_test, y_test, clf, vec, training_data, testing_data):
    training_text = [x[0] for x in training_data]
    training_labels = [x[1] for x in training_data]
    testing_text = [x[0] for x in testing_data]
    testing_labels = [x[0]for x in testing_data]
    #one_coef12, two_coef12, three_coef12 = get_top_coefficients(clf12, vec12, n=5)
    uncharacteristic, characteristic = get_top_coefficients(clf, vec, n=5)
    print('top coefs for class 0: %s' % str(uncharacteristic))
    print('top coefs for class 1: %s' % str(characteristic))
    #print('top coefs for class 3: %s' % str(three_coef12))
    # enter something here to collect data from the user about which words to delete and add this functionality into
    # train after removing features
    #new_clf = methods.train_after_removing_features(X.copy(), y, vec, ['how', 'will'])
    #print('testing accuracy=%.5g' % accuracy_score(y12_test, new_clf.predict(X12_test)))
    errors = get_top_errors(X_test, y_test, clf, testing_text)
    
    un_idx = most_predictive_term_in_doc(X_test[0], clf, 0)
    char_idx = most_predictive_term_in_doc(X_test[0], clf, 1)
    #three_idx = most_predictive_term_in_doc(X12[2], new_clf, 0)
    # this doesn't seem to work for a 3-class classifier
    print('for document %s, the term most predictive of the uncharacteristic class is %s (index=%d)' % (testing_text[0], vec.get_feature_names()[one_idx], one_idx))
    print('for document %s, the term most predictive of the characteristic class is %s (index=%d)' % (testing_text[0], vec.get_feature_names()[two_idx], two_idx))
    #print('for document %s, the term most predictive of class 3 is %s (index=%d)' %
    #      (training_text_12[0], vec12.get_feature_names()[three_idx], three_idx))
    print(testing_text[0], "is classified as ", training_labels[0])
    print_errors(errors, clf, X_test, vec, window=10)

def get_top_errors(X_test, y_test, clf, filenames, n=10):
    """
    Use clf to predict the labels of the testing data in X_test. 
    We want to find incorrectly predicted documents. Furthermore, we want to look at those 
    where the probability of the incorrect label, according to the classifier, is highest.
    Use the .predict_proba method of the classifier to get the probabilities of
    each class label. Return the n documents that were misclassified, sorted by the
    probability of the incorrect label. The returned value is a list of dicts, defined below.
    Params:
        X_test......the testing matrix
        y_test......the true labels for each testing document
        filenames...the filenames for each testing document
        clf.........a trained LogisticRegression object
        n...........the number of errors to return
    Returns:
        A list of n dicts containing the following key/value pairs:
           index: the index of this document (in the filenames array)
           probas: a numpy array containing the probability of class 0 and 1
           truth: the true label
           predicted: the predicted label
           filename: the path to the file for this document
    """
    dicts = []
    wrong = []
    predicted = clf.predict(X_test)
    probas = clf.predict_proba(X_test)

    # find all features where predicted label does not match true label
    # add all output attributes to a list of wrong features
    for i in range(len(predicted)):
        #for j in range(0,2):
        if predicted[i] != y_test[i]:
            wrong.append((i, probas[i], y_test[i], predicted[i], filenames[i]))
    
    # sort list of wrong features by highest probability that predicted label is not true label
    most_wrong = sorted(wrong, key=lambda x: x[4], reverse=True)

    # create dictionary object for top n errors and add to output list of dictionaries
    for j in range(n):
        obj = {}
        obj['index'] = most_wrong[j][0]
        obj['probas'] = most_wrong[j][1]
        obj['truth'] = most_wrong[j][2]
        obj['predicted'] = most_wrong[j][3]
        obj['text'] = most_wrong[j][4]
        dicts.append(obj)   
    return dicts

def index_of_term(vec, term):
    """ This returns the column index corresponding to this term."""
    return vec.get_feature_names().index(term)

def train_after_removing_features(X, y, vec, features_to_remove):
    """
    Set to 0 the columns of X corresponding to the terms in features_to_remove. 
    Then, train a new classifier on X and y and return the result.
    Params:
        X....................the training matrix
        y....................the true labels for each row in X
        features_to_remove...a list of strings (entries in the vocabulary) that
                             should be removed from X
    Returns:
       The classifier fit on the modified X data.
    """

    indices_of_features_to_remove = []
    for f in features_to_remove:
        indices_of_features_to_remove.append(index_of_term(vec, f))
    
    X_copy = X.copy()
    index_array = X_copy.indices

    for i in indices_of_features_to_remove:
        for x in range (0, len(index_array)):
            if index_array[x] == i:
                X_copy.data[x] = 0
        
    model = get_clf()
    model.fit_transform(X_copy, y)
    return model

def find_characteristic_tweets_with_mentions(all_tweets, tweet_codes, scale_item):
    positive_mentions = []
    for full_tweet in all_tweets:
        for coded_tweet in tweet_codes:
            if coded_tweet[0] == full_tweet[0] and coded_tweet[scale_item] == '3' and '@' in full_tweet[3]:
                # tweet is characteristic for scale item and contains at least one mention
                positive_mentions.append(full_tweet)
            
    return positive_mentions

def find_two_sided_conversations_without_retweets(tweet_dictionary, positive_mentions):
    
    to_from_users = []
    for tweet in positive_mentions:
        text = tweet[3]
        mentions = re.findall('@(.*?) ', text, re.DOTALL)
        to_from_users.append((tweet[6], mentions))
        
    characteristic_conversations = []
    
    for item in to_from_users:
        user_from = item[0]
        for user_mentioned in item[1]:
            conversation = []
            for tweet in tweet_dictionary:
                if (tweet['from_user_name'] == user_from and user_mentioned in tweet['text']) or (tweet['from_user_name'] == user_mentioned and user_from in tweet['text']):
                    # do not include retweets
                    if ('RT' not in tweet['text']):
                        conversation.append(tweet)
        one_sided = True
        for t in conversation:
            if t['from_user_name'] != user_from:
                one_sided = False
        # only include conversations where more than one user is sending tweets
        if not one_sided:
            characteristic_conversations.append(conversation)
        
    return characteristic_conversations

def find_conversations(tweet_dictionary, positive_mentions):
    
    to_from_users = []
    for tweet in positive_mentions:
        text = tweet[3]
        mentions = re.findall('@(.*?) ', text, re.DOTALL)
        to_from_users.append((tweet[6], mentions))
        
    characteristic_conversations = []
    
    for item in to_from_users:
        user_from = item[0]
        for user_mentioned in item[1]:
            conversation = []
            for tweet in tweet_dictionary:
                if (tweet['from_user_name'] == user_from and user_mentioned in tweet['text']) or (tweet['from_user_name'] == user_mentioned and user_from in tweet['text']):
                    # do not include retweets
                    if ('RT' not in tweet['text']):
                        conversation.append(tweet)
        characteristic_conversations.append(conversation)
        
    return characteristic_conversations

def compute_number_of_tweets_in_conversations(conversation_list):
    number_of_tweets = 0
    for user in conversation_list:
        number_of_tweets += len(user) 
    return number_of_tweets

def sort_and_count_conversations(conversation_list):
    sorted_and_counted = []
    for item in conversation_list:
        number_of_conversations = []
        for convo in item:
            number_of_conversations.append(len(convo))
        sorted_conversation_counts = sorted(number_of_conversations)
        sorted_and_counted.append(sorted_conversation_counts)
    return sorted_and_counted

def find_and_view_conversations(conversation_list, upper_threshold, lower_threshold):
    long_conversations = []
    for item in conversation_list:
        if upper_threshold > len(item) > lower_threshold:
            long_conversations.append(item)
    
    print('number of conversations: %d' % len(long_conversations))
    
    for item in long_conversations:
        print("\nNEW CONVERSATION ----------------------------------------------------------------------------- \n\n")
        for tweet in item:
            print(tweet['created_at'])
            print("FROM: ", tweet['from_user_name'], "  ", tweet['text'], '\n')

def sort_and_count_conversations(conversation_list):
    sorted_and_counted = []
    for item in conversation_list:
        number_of_conversations = []
        for convo in item:
            number_of_conversations.append(len(convo))
        sorted_conversation_counts = sorted(number_of_conversations)
        sorted_and_counted.append(sorted_conversation_counts)
    return sorted_and_counted



