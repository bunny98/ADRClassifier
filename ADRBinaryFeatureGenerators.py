import string
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import numpy as np
from sklearn import svm
from collections import defaultdict
from featureextractionmodules.FeatureExtractionUtilities import FeatureExtractionUtilities
import pickle
import pandas as pd

stemmer = PorterStemmer()


def loadFeatureExtractionModuleItems():
    '''
        Load the various feature extraction resources
    '''
    FeatureExtractionUtilities.loadItems()

def loadData(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
    '''
    loaded_data_set = defaultdict(list)
    infile = open(f_path)
    for line in infile:
        line = line.decode('utf8', 'ignore').encode('ascii', 'ignore')
        try:
            items = line.split('\t')
            if len(items) > 3:
                tweet_id =  items[0]
                user_id = items[1]
                text = string.lower(string.strip(items[-1]))
                class_ = items[2]

                senttokens = text.split()  #nltk.word_tokenize(_text)
                stemmed_text = ''
                for t in senttokens:
                    stemmed_text += ' ' + stemmer.stem(t)


                loaded_data_set['id'].append(tweet_id + '-' + user_id)
                loaded_data_set['synsets'].append(FeatureExtractionUtilities.getSynsetString(text, None))
                loaded_data_set['clusters'].append(FeatureExtractionUtilities.getclusterfeatures(text))
                loaded_data_set['text'].append(stemmed_text)
                loaded_data_set['unstemmed_text'].append(text)
                loaded_data_set['class'].append(class_)

        except UnicodeDecodeError:
            print 'please convert to correct encoding..'

    infile.close()
    return loaded_data_set

if __name__ == '__main__':
    #LOAD THE FEATURE EXTRACTION RESOURCES
    loadFeatureExtractionModuleItems()

    #LOAD THE DATA -- *SAMPLE SCRIPT USES THE SAME DATA FOR TRAINING AND TESTING*
    # data_set_filename = 'adr_classify_twitter_data_downloaded.txt'
    training_data = loadData('train.txt')
    testing_data = loadData('test.txt')


    #GENERATE THE TRAINING SET FEATURES
    print 'GENERATING TRAINING SET FEATURES.. '
    training_data['sentiments'] = FeatureExtractionUtilities.getsentimentfeatures(training_data['unstemmed_text'])
    training_data['structuralfeatures'] = FeatureExtractionUtilities.getstructuralfeatures(training_data['unstemmed_text'])
    training_data['adrlexicon'] = FeatureExtractionUtilities.getlexiconfeatures(training_data['unstemmed_text'])
    training_data['topictexts'], training_data['topics'] = FeatureExtractionUtilities.gettopicscores(training_data['text'])
    training_data['goodbad'] = FeatureExtractionUtilities.goodbadFeatures(training_data['text'])

    #SCALE THE STRUCTURAL FEATURES
    scaler1 = preprocessing.StandardScaler().fit(training_data['structuralfeatures'])
    train_structural_features = scaler1.transform(training_data['structuralfeatures'])

    #INITIALIZE THE VARIOUS VECTORIZERS
    vectorizer = CountVectorizer(ngram_range=(1,3), analyzer = "word", tokenizer = None, preprocessor = None, max_features = 5000)
    synsetvectorizer = CountVectorizer(ngram_range=(1,1),analyzer="word",tokenizer=None,preprocessor=None,max_features = 2000)
    clustervectorizer = CountVectorizer(ngram_range=(1,1),analyzer="word",tokenizer=None,preprocessor=None,max_features = 1000)
    topicvectorizer = CountVectorizer(ngram_range=(1,1),analyzer="word",tokenizer=None,preprocessor=None,max_features=500)

    #FIT THE TRAINING SET VECTORS
    print 'VECTORIZING TRAINING SET FEATURES.. '
    training_data_vectors = vectorizer.fit_transform(training_data['text']).toarray()
    train_data_synset_vector = synsetvectorizer.fit_transform(training_data['synsets']).toarray()
    train_data_cluster_vector = clustervectorizer.fit_transform(training_data['clusters']).toarray()
    train_data_topic_vector = topicvectorizer.fit_transform(training_data['topictexts']).toarray()
    
    #CONCATENATE THE TRAINING SET VECTORS
    training_data_vectors = np.concatenate((training_data_vectors, train_data_synset_vector), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, training_data['sentiments']), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, train_data_cluster_vector), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, train_structural_features), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, training_data['adrlexicon']), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, training_data['topics']), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, train_data_topic_vector), axis=1)
    training_data_vectors = np.concatenate((training_data_vectors, training_data['goodbad']), axis=1)
 
    
    #GENERATE THE TEST SET FEATURES
    print 'GENERATING TEST SET FEATURES.. '
    testing_data['sentiments'] = FeatureExtractionUtilities.getsentimentfeatures(testing_data['unstemmed_text'])
    testing_data['structuralfeatures'] = FeatureExtractionUtilities.getstructuralfeatures(testing_data['unstemmed_text'])
    testing_data['adrlexicon'] = FeatureExtractionUtilities.getlexiconfeatures(testing_data['unstemmed_text'])
    testing_data['topictexts'],testing_data['topics'] = FeatureExtractionUtilities.gettopicscores(testing_data['text'])
    testing_data['goodbad'] = FeatureExtractionUtilities.goodbadFeatures(testing_data['text'])

    #TRANSFORM THE TEST SET STRUCTURAL FEATURES
    test_structural_features = scaler1.transform(testing_data['structuralfeatures'])

    #TRANSFORM THE TEST SET VECTORS
    print 'VECTORIZING TEST SET FEATURES.. '
    test_data_vectors = vectorizer.transform(testing_data['text']).toarray()
    test_data_synset_vectors = synsetvectorizer.transform(testing_data['synsets']).toarray()
    test_data_cluster_vectors = clustervectorizer.transform(testing_data['clusters']).toarray()
    test_data_topic_vectors = topicvectorizer.transform(testing_data['topictexts']).toarray()

    #CONCATENATE THE TEST SET VECTORS
    test_data_vectors = np.concatenate((test_data_vectors, test_data_synset_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, testing_data['sentiments']), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, test_data_cluster_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, test_structural_features), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, testing_data['adrlexicon']), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, testing_data['topics']), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, test_data_topic_vectors), axis=1)
    test_data_vectors = np.concatenate((test_data_vectors, testing_data['goodbad']), axis=1)
    

    #TRAIN THE SVM CLASSIFIER
    print 'TRAINING THE CLASSIFIER WITH THE FOLLOWING PARAMETERS: '

    f_scores = {}
    c = 140
    w = 3
    print 'C = ' + str(c) + ', ' + 'positive class weight ' + ' = ' + str(w)+'\n'

    svm_classifier = svm.SVC(C=c, cache_size=200, class_weight={'1':w,'0':1}, coef0=0.0, degree=3,
                             gamma='auto', kernel='rbf', max_iter=-1, probability=True, random_state=None,
                             shrinking=True, tol=0.001, verbose=False)
    svm_classifier = svm_classifier.fit(training_data_vectors, training_data['class'])

    #SAVING THE MODEL
    print 'SAVING THE TRAINED MODEL'
    filename = 'SVM_CLASSIFIER.sav'
    pickle.dump(svm_classifier, open(filename, 'wb'))

    #MAKE PREDICTIONS ON THE TEST SET
    print 'MAKING PREDICTIONS ON THE TEST SET..\n'
    result = svm_classifier.predict(test_data_vectors)
    test_gold_classes = testing_data['class']
    
    #COMPUTE THE ADR F-SCORE
    print 'PERFORMANCE METRICS:\n'
    try:
        tp=0.0
        tn=0.0
        fn=0.0
        fp=0.0
        for pred,gold in zip(result,test_gold_classes):
            if pred == '1' and gold == '1':
                tp+=1
            if pred == '1' and gold == '0':
                fp+=1
            if pred == '0' and gold == '0':
                tn +=1
            if pred == '0' and gold == '1':
                fn+=1
        adr_prec = tp/(tp+fp)
        adr_rec = tp/(tp+fn)
        fscore = (2*adr_prec*adr_rec)/(adr_prec + adr_rec)
        print 'Precision for the ADR class .. ' + str(adr_prec)
        print 'Recall for the ADR class .. ' + str(adr_rec)
        print 'ADR F-score .. ' + str(fscore)
        f_scores[str(c)+'-'+str(w)] = fscore
    except ZeroDivisionError:
        print 'There was a zerodivisionerror'
        print 'Precision for the ADR class .. ' + str(0)
        print 'Recall for the ADR class .. ' + str(0)
        print 'ADR F-score .. ' + str(0)
