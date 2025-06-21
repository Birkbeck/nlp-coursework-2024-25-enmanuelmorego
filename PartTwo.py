# Libraries
from pathlib import Path
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from nltk import word_tokenize
import re
from transformers import BertTokenizerFast


def read_speeches_csv(path=Path.cwd() / "texts" / "p2-texts"):
    '''
    Function to load csv files into pandas data frames

    Args:
        Function defaults to a specific location to search for the files unless otherwise specified

    Returns
        Pandas data frame
    '''
    # Extract file name
    file = os.listdir(path)[0]
    file_load = os.path.join(path, file)

    # Read data
    df = pd.read_csv(file_load)
    return df

def speeches_clean(df):
    '''
    Function that takes a data frame containing speeches, and performs custom cleaning tasks on it
    Custom cleaning tasks are:
        - Column 'party': replaces all entries 'Labour (Co-op)' with 'Labour'
        - Column 'party': removes all values where entry is 'Speaker'
        - Column 'party': only keeps the rows of the four most common parties
            Find the frequency count for each party, and keep the top 4 only
        - Column 'speech_class': removes all rows where value is NOT 'Speech'
        - Column 'speech': removes any entries where the length of the speech is less than 1000 characters

    Args: 
        df: Pandas data frame

    Returns:
        A Pandas data frame, cleaned
    '''
    # (a).i Clean Labour (Co-op) values
    df_cleaned = df.replace('Labour (Co-op)', 'Labour')

    # (a).ii Remove rows where 'party' == 'Speaker'
    '''Note: Remove speaker rows first, otherwise this will interfere with finding the most common parties'''
    df_cleaned = df_cleaned[df_cleaned['party'] != 'Speaker']

    # (a).ii Remove rows where the value of 'party' is not one of the 4 most common parties
    parties_count = df_cleaned['party'].value_counts().sort_values(ascending=False)
    # # Extract the name of the 4 most common parties 
    top4_parties = parties_count.index[:4].tolist()
    # # Filter to top 4 most common parties
    df_cleaned2 = df_cleaned[df_cleaned['party'].isin(top4_parties)]

    # (a).iii Remove rows where value in 'speech_class' is not 'Speech
    df_cleaned2 = df_cleaned2[df_cleaned2['speech_class'] == 'Speech']

    return df_cleaned2

def ml_pipeline(**kwargs):
    '''
    Function which processes and build ML models given the speeches data and prepares the data to be fed into ML models:
    The pipeline:
        Splits into train, test sets
        Vectorises the data
        Trains a RandomForest Model
        Trains a Linear SVM classifer
        Extracts the CLassification Report for each model
        Macro-Average F1 Score
    
    Arguments can be passed as key value pairs. Some arguments are mandatory whilst other are optionals. When optional arguments are not provided
    the function will use defaul values
    Ars:
        data (mandatory): A cleaned pandas data frame
        ngram (optional): a tuple containing the ngram to consider to pass in the TfidVectorizer function
            default value: (1,1) unigrams
        stop_words (optional): A string containing the value for the stop_words argument for TfidVectorizer. If set ti 'english', stop words would be removed
            default value: None - Stop words would not be removed

    '''
    # Extract input parameters
    input_dict = kwargs

    # Extract data from input
    df = input_dict.get('data')
    ngram = input_dict.get('ngram', (1,1))
    stop_words = input_dict.get('stop_words', None)
    tokenizer = input_dict.get('tokenizer', None)

    # Tokenizer print: 
    if tokenizer is not None:
        token_print = tokenizer.__name__
    else:
        token_print = tokenizer
    print("\nArguments:")
    print(f"\tNgram: {ngram}\n\tStop words: {stop_words}\n\tTokenizer: {token_print}\n")

    # (b) Generate object that splits data using stratified sampling, and random seed of 26
    splitter_obj = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 26) 
    # Split data
    for train_index, test_index in splitter_obj.split(df, df['party']):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

    # (b) Split target in both training and testing set
    y_train, y_test = train['party'], test['party']

    # (b) Create vectorised data for x objects
    '''
    Max features set to 3000
    stop_words, ngram = defined by parameters when function is called
    '''
    vectorizer = TfidfVectorizer(max_features = 3000, 
                                 stop_words=stop_words, 
                                 ngram_range = ngram,
                                 tokenizer = tokenizer)
    x_train = vectorizer.fit_transform(train['speech'])
    x_test = vectorizer.transform(test['speech'])

    # (c) Train random forest
    random_forest = RandomForestClassifier(n_estimators=10, n_jobs = -1) # TODO Set to small number for training, bring back to 300 for real testing
    random_forest.fit(x_train, y_train)
    random_forest_y_predict = random_forest.predict(x_test)

    # (c) Train SVM
    svm = LinearSVC()
    svm.fit(x_train, y_train)
    svm_y_predict = svm.predict(x_test)

    # Get label names
    target_names = y_test.unique()

    # Results section 
    print(f"{"="*20} Random Forest Performance {"="*20}")
    rf_cr = classification_report(y_test, random_forest_y_predict, target_names = target_names, output_dict = True)
    print(classification_report(y_test, random_forest_y_predict, target_names = target_names))

    print(f"{"="*20} SVC Performance {"="*20}")
    svc_cr = classification_report(y_test, svm_y_predict, target_names = target_names, output_dict = True)
    print(classification_report(y_test, svm_y_predict, target_names = target_names))

    return {'rf': rf_cr, 'svc': svc_cr}

def my_tokenizer_basic(text):
    '''
    Basic tokenizer that keeps stop words in
    '''
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalpha()]

# Building/load tokenizer 
bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def my_tokenizer_bert(text):
    '''
    Tokenizer using BERT fast tokenizer mode
    '''
    # Clean the text. Remove special characters, such as \n, \t etc and extra white spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    tokenized_text = bert_tokenizer.tokenize(text)
    return tokenized_text


def data_pre_processing(input_dict=None, **kwargs):
    '''
    Function to pre process the data in preparation for ML models.
        Splits into train, test sets
        Vectorises the data
        Returns 6 processed objects

    For flexibility, the input of the function is defined as a dicionary or key value pairs. This allows user to add multiple parameters, or if no parameters are given
    the function can simply use pre-set default values
    
    Args:
        data (mandatory) = a pandas data frame to process
        ngram (optional) = a tuple containing the ngram to consider to pass in the TfidVectorizer function
    
    '''
    # Check whether input is a dicitonary or key-value pairs
    if input_dict is None:
        input_dict = kwargs

    # Extract data from input
    df = input_dict.get('data')
    # Extract ngram range, if not provided, default to (1,1)
    ngram = input_dict.get('ngram', (1,1))
    # Extract stop words argument
    stop_words = input_dict.get('stop_words', None)

    # Generate object that splits data
    splitter_obj = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 26) 
    # Split data
    for train_index, test_index in splitter_obj.split(df, df['party']):
        train = df.iloc[train_index]
        test = df.iloc[test_index]

    # Split target in both training and testing set
    y_train, y_test = train['party'], test['party']

    # Create vectorised data for x objects
    '''Prompt indicated to use all default parameters, except for ommitting English stop words and max_featrues which needs to be set to 3000'''
    vectorizer = TfidfVectorizer(max_features = 3000, stop_words=stop_words, ngram_range = ngram)
    x_train = vectorizer.fit_transform(train['speech'])
    x_test = vectorizer.transform(test['speech'])

    # Return objects
    return x_train, x_test, y_train, y_test, train['speech'], test['speech']


def ml_models(input_dict=None, **kwargs):
    '''
    Pipeline to train and run ML models
    
    The function also calls the pre_processing function which prepares the data

    For flexibility, the input of the function is defined as a dicionary or key value pairs. This allows user to add multiple parameters, or if no parameters are given
    the function can simply use pre-set default values.

    This is useful as it will allow the user to adjust hyperparameters as needed.
    
    Args:
        data (mandatory) = a pandas data frame to process
        ngram (optional) = a tuple containing the ngram to consider to pass in the TfidVectorizer function (this is passed into the data_pre_processing function)

    '''    
     # Check whether input is a dicitonary or key-value pairs
    if input_dict is None:
        input_dict = kwargs

    # Extract data from input
    df = input_dict.get('data')
    # Extract ngram range, if not provided, default to (1,1)
    ngram = input_dict.get('ngram', (1,1))
    # Extract stop words argument
    stop_words = input_dict.get('stop_words', None)

    # Prepare data for ML
    x_train, x_test, y_train, y_test, train_text, test_text = data_pre_processing(data = df, ngram= ngram, stop_words=stop_words)

    # Train random forest
    random_forest = RandomForestClassifier(n_estimators=300, n_jobs = -1)
    random_forest.fit(x_train, y_train)
    random_forest_y_predict = random_forest.predict(x_test)

    # Train SVM
    svm = LinearSVC()
    svm.fit(x_train, y_train)
    svm_y_predict = svm.predict(x_test)

    # Get label names
    target_names = y_test.unique()

    # Results section 
    print(f"{"="*20} Random Forest Performance {"="*20}")
    rf_cr = classification_report(y_test, random_forest_y_predict, target_names = target_names, output_dict = True)
    print(classification_report(y_test, random_forest_y_predict, target_names = target_names))

    print(f"{"="*20} SVC Performance {"="*20}")
    svc_cr = classification_report(y_test, svm_y_predict, target_names = target_names, output_dict = True)
    print(classification_report(y_test, svm_y_predict, target_names = target_names))

    return {'rf': rf_cr, 'svc': svc_cr}

    # def my_tokenizer_simple()
   

if __name__ == "__main__":
    
    '''Section A'''
    print("\n\n\t\t==== Section A ====\n")
    print(f"Data load and initial cleaning...")
    
    # Load speeches data frame
    df = read_speeches_csv()
    # Clean data frame
    df_cleaned = speeches_clean(df)
    print(f"Dimensions of cleaned speeches data frame")
    print(df_cleaned.shape)

    '''Section B'''
    print("\n\n\t\t==== Section B ====\n")
    print(f"Data preprocessing for ML models - see function data_pre_processing...")

    print("\n\n\t\t==== Section C ====\n")
    print(f"Trains and compare performance of Random Forest vs SVM with a linear kernel...\n")

    # Dictionary to record the Macro Avg F1 score for each tested model
    f1_results = {}

    print(f"\nModel 1:\n\tTfidvectorizer unigrams only")
    section_c = ml_pipeline(data = df_cleaned, stop_words = 'english')
    # Save results into a dictionary
    f1_results['f1_ma_rf_unigram'] =  round(section_c['rf']['macro avg']['f1-score'] ,2)
    f1_results['f1_ma_svc_unigram'] = round(section_c['svc']['macro avg']['f1-score'], 2)

    print("\n\n\t\t==== Section D ====\n")

    print(f"\nModel 2:\n\tTfidvectorizer unigrams, bi-grams and tri-grams")
    section_d = ml_pipeline(data = df_cleaned, ngram = (1,3), stop_words = 'english')
    # Save results into a dictionary
    f1_results['f1_ma_rf_uni_bi_trigrams'] =  round(section_d['rf']['macro avg']['f1-score'] ,2)
    f1_results['f1_ma_svc_uni_bi_trigrams'] = round(section_d['svc']['macro avg']['f1-score'], 2)

    print("\n\n\t\t==== Section E ====\n")
    print(f"Try a few custom tokenizers and compare performance")

    print(f"\nModel 1:\n\tBasic word tokenizer, keep english stop words")
    section_e_basic_t =  ml_pipeline(data = df_cleaned, ngram = (1,3), tokenizer = my_tokenizer_basic)
    f1_results['f1_ma_rf_uni_bi_trigrams_basictoken'] =  round(section_e_basic_t['rf']['macro avg']['f1-score'] ,2)
    f1_results['f1_ma_svc_uni_bi_trigrams_basictoken'] = round(section_e_basic_t['svc']['macro avg']['f1-score'], 2)

    print(f"\nModel 2:\nBERT word tokenizer, keep english stop words")
    
    # Create tokenizer
    # section_e_bert_t =  ml_pipeline(data = df_cleaned, ngram = (1,3), tokenizer = my_tokenizer_bert)
    # f1_results['f1_ma_rf_uni_bi_trigrams_berttoken'] =  round(section_e_bert_t['rf']['macro avg']['f1-score'] ,2)
    # f1_results['f1_ma_svc_uni_bi_trigrams_berttoken'] = round(section_e_bert_t['svc']['macro avg']['f1-score'], 2)


    print(".............")
    print(f1_results)


    

    '''
    F1 Score: 2 / ((1 / precision) + (1 / recall)) is a combination of the precision and recall metrics
        Precision: TP / (TP + FP) the proportion of all of the items predicted as positive that were actually positive 
        Recall: TP/(TP + FN) the proportion of all of the actual positive values that the model correctly identified as positive (it accounts for false negatives)

    F1 scores are particularly useful with imbalanced data as it takes into account the type of mistakes the model makes. This means that for the model to get a good 
    F1 score (close to 1) it has to achieve both high precision and recall. 
    
    For example, if the model labels objects as'yes' most of the time, and the dataset has mostly
    'yes' entries, the model might achieve high precision but low recall (as it missed the few no cases). The F1 scores takes this into account and this gap between the metrics
    would mean a low F1 score. So the higher the F1 score, the more confident the user can be that the model correctly predicted the values irrespective of the proportion
    of such values in the data set 
    '''
