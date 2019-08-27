import pickle            # Loading and saving objects with pickle
import numpy as np
import string
from pycorenlp import StanfordCoreNLP
import re

# ------------------------------------------
# -------- UTILITIES FROM NLTK ----------
import nltk
from nltk.stem import WordNetLemmatizer                     # stemmer lemmatizer
from nltk import pos_tag                                    # pos tagging used in preprocessing done in clean func
from nltk.tokenize import word_tokenize                     # using this in clean func as well
from nltk.corpus import wordnet                             # Cleaner uses this
from nltk.corpus import stopwords                           # For removing stopwords
from nltk import sent_tokenize
from nltk.tag import CRFTagger                              # POS tagger
from nltk.corpus import brown                               # using in the HMM POS tagger

# ------------------------------------------
# -------- UTILITIES FROM SKLEARN ----------
from sklearn_crfsuite import CRF

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

# ------------------------------------------
# -------------- CLASSIFIERS ---------------
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier


# ------------------------------------------
#  PICKLE FUNCTIONS FOR SAVING AND LOADING OBJECTS FROM FILES

def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# ------------------------------------------
# LOADING THE TRAIN AND TEST DATASETS
# datasets are being loaded in the form of data frames with 3 columns: character, dialogue, action
# character column contains name of the character
# dialogue column contains the text of the utterance
# action column contains actions associated with the character and the utterance

train_df = load_obj("train_combined_df")
test_df = load_obj("test_combined_df")

# ---------------------------------------------------
# FUNCTIONS USED FOR FURTHER CLEANING OF THE PROCESSED TEXT

def get_wordnet_pos(treebank_tag):
    """
    Function used for converting the treebank POS tags to wordnet tags, so that they become understandable for the lemmatizer.

    :param treebank_tag: tag from the treebank
    :return: wordnet tag
    """

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None  # for easy if-statement

def decontracted(phrase):
    """
    Function used for solving the contractions like can't or won't.

    :param phrase: phrase that may contain the contraction that should be solved
    :return: modified version of the sentenced passed as a parameter that contains decontracted version of the contraction
    """

    # specific
    phrase = re.sub(r"won\'t|won\’t", "will not", phrase)
    phrase = re.sub(r"can\'t|can\’t", "cannot", phrase)
    phrase = re.sub(r"'cause|’cause", "because", phrase)
    phrase = re.sub(r"c'mon|c’mon", "come on", phrase)
    phrase = re.sub(r"c'mere|c’mere", "come here", phrase)
    phrase = re.sub(r"'course|’course", "of course", phrase)
    phrase = re.sub(r"gimme", "give me", phrase)
    phrase = re.sub(r"gonna", "going to", phrase)
    phrase = re.sub(r"gotta", "got to", phrase)
    phrase = re.sub(r"let's|let’s", "let us", phrase)
    phrase = re.sub(r"'tis|’tis", "it is", phrase)
    phrase = re.sub(r"'twas|’twas", "it was", phrase)
    phrase = re.sub(r"y'all|y’all", "you all", phrase)

    # general
    phrase = re.sub(r"n\'t|n\’t", " not", phrase)
    phrase = re.sub(r"\'re|’re", " are", phrase)
    phrase = re.sub(r"\'s|’s", " is", phrase)
    phrase = re.sub(r"\'d|’d", " would", phrase)
    phrase = re.sub(r"\'ll|’ll", " will", phrase)
    phrase = re.sub(r"\'t|’t", " not", phrase)
    phrase = re.sub(r"\'ve|’ve", " have", phrase)
    phrase = re.sub(r"\'m|’m", " am", phrase)
    return phrase


def lemmatize_action(text):
    """
    Function used for lemmatizing words in the given text.

    :param text: text that is supposed to be lemmatized
    :return: lemmatized version of the initial text
    """

    lemmatiser = WordNetLemmatizer()
    tokens = word_tokenize(text)  # Generate list of tokens
    tokens_pos = pos_tag(tokens)
    text2 = ' '

    for word, tag in tokens_pos:
        wntag = get_wordnet_pos(tag)
        if wntag is None:  # not supply tag in case of None
            lemma = lemmatiser.lemmatize(word)
        else:
            lemma = lemmatiser.lemmatize(word, pos=wntag)
        text2 += lemma + ' '
    return text2


# Regular expressions for removing and replacing bad symbols that may occur in the text used in the clean_text function
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))



def clean_text(text):
    """
    Function used for additional cleaning of the text.

    :param text: text that is supposed to be cleaned
    :return: cleaned version of the initial text
    """

    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwords from text
    text = decontracted(text) # decontracted text

    # Lemmatization of the words in the text
    text2 = lemmatize_action(text)

    return text2


# ---------------------------------------------------
# FUNCTIONS USED FOR ANALYSIS OF THE TEXT WITH THE USE OF DEPENDENCY GRAMMAR

def StanfordCoreNLPAnnotate(text):
    """
    Function used for generating dependency and context free grammar trees for the text passed as a parameter.

    :param text: text that should be annotated
    :return: annotated text
    """
    nlp = StanfordCoreNLP('http://localhost:9000')

    annotated_text = nlp.annotate(text, properties={
        'annotators': 'parse',
        'outputFormat': 'json'
    })
    return annotated_text


def DependencyParseTree(annotated_text):
    """
    Function used for extracting the features of the text passed as a parameter in the form of the statistics generated
    from the dependency trees build from the text. (Text can contain more than one sentence in this case the average of
    the values of all the sentences is being returned.)

    :param annotated_text: annotated text for which the features should be extracted
    :return:
        avg_depth - average depth of the trees from all the sentences in the text,
        avg_branching_factor - average branching factor from all the sentences in the text,
        avg_total_branches - average number of all the branches from all the sentences in the text,
        tag_cnt - dictionary containing the counts of the appearances of different elements of the dependency grammar from
        all the sentences in the text
    """
    depth_list = []
    branching_factor_list = []
    total_branches_list = []
    tag_cnt = []
    for sentence in annotated_text['sentences']:
        # Processing dependancy tree
        parse_tree = sentence['basicDependencies']
        depth = 0
        max_branching_factor = 0
        total_branches = len(parse_tree)
        dependent_list = []

        # Find and process the root
        for node in parse_tree:
            if node['dep'] == 'ROOT':
                dependent_list.append(node['dependent'])
                tag_cnt.append(node['dep'])
                parse_tree.remove(node)
                depth += 1

        temp_dependent_list = []
        parse_tree_copy = []
        # Remove parent nodes until nothing left
        while parse_tree != []:
            temp_branching_factor = 0
            parse_tree_copy[:] = parse_tree
            # Find and process all children of current parent
            for node in parse_tree_copy:
                if node['governor'] in dependent_list:
                    temp_branching_factor += 1
                    # Set new parents
                    temp_dependent_list.append(node['dependent'])
                    tag_cnt.append(node['dep'])
                    # Remove parent nodes
                    parse_tree.remove(node)
            dependent_list = temp_dependent_list
            temp_dependent_list = []
            depth += 1
            # Keep track of maximum branching factor
            if temp_branching_factor > max_branching_factor:
                max_branching_factor = temp_branching_factor

        # Save stats for this sentence
        depth_list.append(depth)
        branching_factor_list.append(max_branching_factor)
        total_branches_list.append(total_branches)

    # Average stats for all sentences in utterance
    avg_depth = np.mean(depth_list)
    avg_branching_factor = np.mean(branching_factor_list)
    avg_total_branches = np.mean(total_branches_list)
    if np.isnan(avg_depth):
        avg_depth = 0
    if np.isnan(avg_branching_factor):
        avg_branching_factor = 0
    if np.isnan(avg_total_branches):
        avg_total_branches = 0


    return avg_depth, avg_branching_factor, avg_total_branches, tag_cnt


# ---------------------------------------------------
# FUNCTIONS USED FOR ANALYSIS OF THE TEXT WITH THE USE OF CONTEXT FREE GRAMMAR

def CFGParseTree(annotated_text):
    """
    Function used for extracting the features of the text passed as a parameter in the form of the statistics generated
    from the context free grammar trees build from the text. (Text can contain more than one sentence in this case the
    average of the values of all the sentences is being returned.)

    :param annotated_text: annotated text for which the features should be extracted
    :return:
        avg_depth - average depth of the trees from all the sentences in the text,
        avg_branching_factor - average branching factor from all the sentences in the text,
        avg_total_branches - average number of all the branches from all the sentences in the text,
    """

    depth_list = []
    branching_factor_list = []
    total_branches_list = []
    for sentence in annotated_text['sentences']:
        parse_tree = sentence['parse']
        # Extract brackets only
        raw_tree = re.sub('[^\(\)]', '', parse_tree)

        # Processing CFG tree
        depth = 0
        max_branching_factor = 0
        total_branches = 0
        # Remove leaf nodes until nothing left
        while raw_tree is not '':
            # Finding leaf nodes
            leaves = re.findall('((\(\))+)', raw_tree)
            for leaf in leaves:
                temp_branching_factor = len(leaf[0]) / 2
                total_branches += temp_branching_factor
                # Keep track of maximum branching factor
                if temp_branching_factor > max_branching_factor:
                    max_branching_factor = temp_branching_factor
            # Remove leaf nodes
            raw_tree = re.sub('((\(\))+)', '', raw_tree)
            depth += 1

        # Ignore root node
        depth -= 1
        total_branches -= 1
        # Save stats for this sentence
        depth_list.append(depth)
        branching_factor_list.append(max_branching_factor)
        total_branches_list.append(total_branches)

    # Average stats for all sentences in utterance
    avg_depth = np.mean(depth_list)
    avg_branching_factor = np.mean(branching_factor_list)
    avg_total_branches = np.mean(total_branches_list)

    if np.isnan(avg_depth):
        avg_depth = 0;
    if np.isnan(avg_branching_factor):
        avg_branching_factor = 0;
    if np.isnan(avg_total_branches):
        avg_total_branches = 0;

    return avg_depth, avg_branching_factor, avg_total_branches


# ---------------------------------------------------
# FUNCTIONS USED FOR ANALYSIS OF THE TEXT WITH THE USE OF POS TAGGING

# NLTK POS TAGGING
def tag_text(text):
    """
    Function used for tagging the given text. This function uses a pos tagger from the NLTK library.

    :param text: text to be associated with the POS tags
    :return: string containing POS tags for the given text; each tag refers to the word at the same index in the sentence
        (eg. for the sentence "My dog likes running around the garden." the returned string with tags is "PRP$ NN VBZ VBG IN DT NN .")
    """

    tokens = nltk.word_tokenize(text)
    token_tag = nltk.pos_tag(tokens)

    tag_list = [x[1] for x in token_tag]
    return ' '.join(tag_list)


# CRF POS TAGGING
# auxiliary functions
def features(sentence, index):
    """
    Function used for generating features for a word. The function takes the indes of the word in a sentence for which the
    features should be extracted, it analysis the structure of the sentence and the structure of the word and returns
    a dictionary with the features.

    :param sentence: sentence used for the analysis given in the form [w1,w2,w3,..],
    :param index: the position of the word in the sentence
    :return: dictionary containing features of the word in the sentence
    """

    return {
        'is_first_capital': int(sentence[index][0].isupper()),
        'is_first_word': int(index == 0),
        'is_last_word': int(index == len(sentence) - 1),
        'is_complete_capital': int(sentence[index].upper() == sentence[index]),
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'is_numeric': int(sentence[index].isdigit()),
        'is_alphanumeric': int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])', sentence[index])))),
        'prefix_1': sentence[index][0],
        'prefix_2': sentence[index][:2],
        'prefix_3': sentence[index][:3],
        'prefix_4': sentence[index][:4],
        'suffix_1': sentence[index][-1],
        'suffix_2': sentence[index][-2:],
        'suffix_3': sentence[index][-3:],
        'suffix_4': sentence[index][-4:],
        'word_has_hyphen': 1 if '-' in sentence[index] else 0
    }

def untag(sentence):
    """
    Function used for extracting only words of the sentence from the tagged data.

    :param sentence: sentence with the tags
    :return: list of words from the sentence
    """
    return [word for word, tag in sentence]

def prepareData(tagged_sentences):
    """
    Function used for preparing the datasets used for training the model. It generates two datasets:
     - one containing the features for all the words in the sentences passed as parameters
     - second one containing tags for all the words in the sentences passed as parameters

    :param tagged_sentences: set with tagged sentences that are used for generation of the datasets
    :return:
        X - list of lists of dictionaries containing features of each word of the each sentence
        y - list of lists of tags for each sentence
    """

    X, y = [], []
    for sentences in tagged_sentences:
        X.append([features(untag(sentences), index) for index in range(len(sentences))])
        y.append([tag for word, tag in sentences])
    return X, y

def generateUtterancesFeatures(sentence):
    """
    Function ued for generating a list of dictionaries containing the features for each word of the given sentence.

    :param sentence: sentence for which words the features should be extracted
    :return: list of dictionaries containing features of the words of the sentence passed as a parameter
    """

    X=[]
    sent = sentence.split()
    X.append([features(sent, index) for index in range(len(sent))])
    return X

# TRAINING THE CRF CLASSIFIER
# Loading the corpus from the treebank for training the pos tagger
tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')
# Preparing the datasets from the corpus for training the classifier
X_train, y_train = prepareData(tagged_sentence)
# Training the CRF classifier for PoS tagging
crf = CRF(
    algorithm='lbfgs',
    c1=0.01,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

def tag_text_CRF(text):
    """
    Function used for tagging the given text. This function uses a CRF pos tagger from the SKLEARN library.

    :param text: text to be associated with the POS tags
    :return: string containing POS tags for the given text; each tag refers to the word at the same index in the sentence
        (eg. for the sentence "My dog likes running around the garden." the returned string with tags is
        "PRON VERB NOUN VERB ADP DET NOUN")
    """

    # Extract features from words in the given text
    features=generateUtterancesFeatures(text)
    # Predict tags for the given utterance
    tags = crf.predict(features)
    return ' '.join(str(t) for w in tags for t in w)


# CRF POS TAGGING - PRE-TRAINED POS-TAGGER
# Path to the pre-trained POS-tagger
TAGGER_PATH = "crfpostagger"
# Initialize tagger
tagger = CRFTagger()
tagger.set_model_file(TAGGER_PATH)

#def tag_text_CRF(text):
#    """
#    Function used for tagging the given text. This function uses a CRF predefined pos tagger.
#
#    :param text: text to be associated with the POS tags
#    :return: string containing POS tags for the given text; each tag refers to the word at the same index in the sentence
#        (eg. for the sentence "My dog likes running around the garden." the returned string with tags is
#        "PRP$ NN VBZ VBG IN DT NN")
#    """
#
#    tags= tagger.tag([word.lower() for word in text.split()])
#    return ' '.join(str(t) for w,t in tags)


# HMM POS TAGGING
Use_HMM = False
if Use_HMM:

    # get the list of all tags
    brown_tags_words = []
    for sent in brown.tagged_sents():
        brown_tags_words.append(("START", "START"))
        brown_tags_words.extend([(tag[:2], word) for (word, tag) in sent])
        brown_tags_words.append(("END", "END"))

    # conditional frequency distribution
    cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
    # conditional probability distribution
    cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)

    brown_tags = [tag for (tag, word) in brown_tags_words]

    cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
    cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)

    distinct_tags = set(brown_tags)


def tag_text_HMM(text):
    """
    Function used for tagging the given text. This function uses a HMM POS tagger.

    :param text: text to be associated with the POS tags
    :return: string containing POS tags for the given text; each tag refers to the word at the same index in the sentence
        (eg. for the sentence "My dog likes running around the garden." the returned string with tags is
        "PP NN VB VB IN AT NN")
    """

    # Remove the punctuation from the given text
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split text into a list of words
    sentence = text.split()

    # Return empty string for an empty text
    sentlen = len(sentence)
    if sentlen == 0:
        return ' '

    viterbi = []
    backpointer = []

    first_viterbi = {}
    first_backpointer = {}
    for tag in distinct_tags:
        # don't record anything for the START tag
        if tag == "START": continue
        first_viterbi[tag] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob(sentence[0])
        first_backpointer[tag] = "START"

    viterbi.append(first_viterbi)
    backpointer.append(first_backpointer)

    for wordindex in range(1, len(sentence)):
        this_viterbi = {}
        this_backpointer = {}
        prev_viterbi = viterbi[-1]

        for tag in distinct_tags:
            # don't record anything for the START tag
            if tag == "START": continue
            best_previous = max(prev_viterbi.keys(),
                                key=lambda prevtag: \
                                    prev_viterbi[prevtag] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(
                                        sentence[wordindex]))
            this_viterbi[tag] = prev_viterbi[best_previous] * \
                                cpd_tags[best_previous].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex])
            this_backpointer[tag] = best_previous

        viterbi.append(this_viterbi)
        backpointer.append(this_backpointer)

    prev_viterbi = viterbi[-1]
    best_previous = max(prev_viterbi.keys(),
                        key=lambda prevtag: prev_viterbi[prevtag] * cpd_tags[prevtag].prob("END"))

    # Best tags sequence (storeed in reverse - will invert later)
    best_tagsequence = ["END", best_previous]
    # Invert the list of backpointers
    backpointer.reverse()

    current_best_tag = best_previous
    for bp in backpointer:
        best_tagsequence.append(bp[current_best_tag])
        current_best_tag = bp[current_best_tag]

    best_tagsequence.reverse()
    best_tagsequence = best_tagsequence[1:-1]

    return ' '.join(str(v) for v in best_tagsequence)

# ----------------------------------------------------------
# WORD2VEC
Use_Glove = True
if Use_Glove:
    # Loading the file with the pre-trained word vectors
    with open("glove.6b/glove.6B.50d.txt", "rb") as lines:
        w2v = {}
        # Preparing a dictionary, where the key is a word and the value is the word's vector
        for line in lines:
            word_vector = []
            for i in line.split()[1:]:
                word_vector.append(float(i.decode("utf-8")))
            w2v[line.split()[0].decode("utf-8")] = word_vector

# -----------------------------------------
# CLASSES USED FOR ORGANIZING THE FEATURES SO THAT THEY CAN BE USED IN THE PIPELINE

class MeanEmbeddingVectorizer(object):
    """
    Class used for transforming words from the datasets into vectors and calculates the mean of the vectors of all the
    words in the utterance.
    """
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = 50 # dimension of the word's vector

    def fit(self, X, y):
        return self

    def transform(self, X):
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        return np.array([
            np.mean([self.word2vec[w] for w in words.split() if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class ItemSelector(BaseEstimator, TransformerMixin):
    """
    Class used picking the data object from the features array as specified by the key.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):
    """
    Class used for extracting text statistics from each utterance for DictVectorizer.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'length': len(text),
                 'num_sentences': text.count('.'),
                 'num_questions': text.count('?'),
                 'num_exclamations': text.count('!')
                 }
                for text in posts]


class TreeStats(BaseEstimator, TransformerMixin):
     """
     Class used for extracting statistics from the parse trees from each utterance for DictVectorizer.
     """

     def fit(self, x, y=None):
        return self

     def transform(self, posts):
        return [{'depth': int(text[0]),
                 'branching_factor': int(text[1]),
                 'total_branches': int(text[2])}
                for text in posts]


class PosDialogueExtractor(BaseEstimator, TransformerMixin):
    """
    Class used for preparing the N-dimensional array of the features used in the various pipelines.
    It takes a dataframe with an action and utterance and produces features from it. Keys for the features defined in
    this function are:
        - 'dialogue',
        - 'dialogue_clean',
        - 'pos_CRF', 'pos', 'pos_HMM'
        - 'tree_stats',
        - 'parse_trees'
        - 'action`.

    This part is being modified based on the features that are selected.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        features = np.recarray(shape=(len(df),),
                               dtype=[('dialogue_clean', object), ('dialogue', object), ('pos_CRF', object), ('pos', object),
                                      ('tree_stats', object), ('action', object), ('pos_HMM', object), ('parse_trees', object)])
        for i, text in enumerate(df.dialogue):
            features['dialogue_clean'][i] = clean_text(text)
            #features['pos'][i] = tag_text(text)
            #features['dialogue'][i] = text
            features['pos_CRF'][i] = tag_text_CRF(text)
            #features['pos_HMM'][i] = tag_text_HMM(text)
            tree = list(DependencyParseTree(StanfordCoreNLPAnnotate(text)))
            #tree = list(CFGParseTree(StanfordCoreNLPAnnotate(text)))
            #features['parse_trees'][i] = str(tree[3])  # Needed for dependency grammar only
            features['tree_stats'][i] = tree[0:3]

        for i, action in enumerate(df.action):
            features['action'][i] = action

        return features


# ------------------------------------------
# CROSS VALIDATION FUNCTION
def crossValidate(dataset, folds):
    """
    Function used for performing the cross-validation on the given dataset. It produces 'folds' number of classifiers and
    picks the best one based on its test accuracy.

    :param dataset: dataset used for cross-validation
    :param folds: number of folds used in the cross-validation
    :return: the best of the classifiers obtained while performing the cross-validation
    """
    kfold = KFold(n_splits=folds, random_state=1, shuffle=True)

    list_of_classifier = []
    for train, test in kfold.split(dataset):
        # kfolds.split gives indices for the dataframe - next we'll be picking entries from the df against those
        train = dataset.iloc[train]
        test = dataset.iloc[test]
        # Splitting the dataset into: x - dialogue and action and y - character
        X_train = train[['dialogue', 'action']]
        y_train = train.character
        X_test = test[['dialogue', 'action']]
        y_test = test.character
        # Calling the pipeline
        pipeline.fit(X_train, y_train)
        # Getting the accuracy for this fold
        y_pred = pipeline.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        # Appending obtained classifier and its accuracy
        list_of_classifier.append([pipeline, accuracy])

    # From the list of classifiers, picking the one with the best accuracy
    accuracy_results = [x[1] for x in list_of_classifier]  # List with only the accuracies from the classifier list
    print(accuracy_results)
    index = accuracy_results.index(max(accuracy_results))  # Picking the classifier with max accuracy
    chosen_classifier = list_of_classifier[index][0]

    return chosen_classifier

# ---------------------------------------
#
def treat_action(df):
    """
    Function used for transforming the dataframe with the utterances. It extracts the character's actions from the
    text of the scene descriptor, saving them in the modified dataframe.

    :param df: dataframe to be modified, containing the character, utterance and scene descriptor
    :return: modified dataframe, containing the character, utterance and the list of character's actions
    """
    for index, row in df.iterrows():
        action = row['action']          # Extract the action
        character = row['character']    # Extract the character

        mod_action = " "
        if action is not []:
            for a in action:
                tokenized_text = sent_tokenize(a)
                for sentence in tokenized_text:
                    if character.lower() in sentence.lower():
                        # Change all the letter in a sentence to lowercase
                        sentence = sentence.lower()
                        # Remove punctuation in the given sentence
                        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
                        # Extract features from words
                        features1 = generateUtterancesFeatures(sentence)
                        # Tag the utterance
                        tags = crf.predict(features1)
                        tags = tags[0]
                        # Lemmatize the sentence
                        sentence = lemmatize_action(sentence)
                        sentence = re.sub(r" \’ ", "\'", sentence)
                        sentence = sentence.split()
                        # Take all the pairs of [char, VERB]
                        for count, word in enumerate(sentence):
                            if word.upper() == character and count < len(tags) - 1:
                                if tags[count + 1] == 'VERB':
                                    mod_action = mod_action + ' ' + sentence[count + 1]

        # Modify the dataframe
        row['action'] = mod_action
    return df

#------------------------------------------
# PIPELINE OBJECT
# Pipeline object which consists of multiple pipelines, each with its own data/transformers. The last pipeline is the classifier
# that is being used.
pipeline = Pipeline([
    # Extract the subject & body
    ('posdialogue', PosDialogueExtractor()),

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's subject line
            #  ('pos', Pipeline([
            #       ('selector', ItemSelector(key='pos')),
            #       ('tfidf', TfidfVectorizer()),
            #   ])),

            # Pipeline for standard bag-of-words model for body
            ('dialogue_bow', Pipeline([
                 ('selector', ItemSelector(key='dialogue_clean')),
                 ('tfidf', CountVectorizer()),
                  # ('best', TruncatedSVD(50))
              ])),

            # Pipeline for pulling ad hoc features from post's body
            #('dialogue_stats', Pipeline([
            #      ('selector', ItemSelector(key='dialogue')),
            #      ('stats', TextStats()),  # returns a list of dicts
            #      ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                  # ('best', TruncatedSVD(50))
            #  ])),

            # Pipeline for glove word2vec
            ('word2vec', Pipeline([
                  ('selector', ItemSelector(key='dialogue_clean')),
                  ('vect', MeanEmbeddingVectorizer(w2v)),  # returns a list of dicts
              ])),

            # Pipeline for pulling features from post's PoS tagging with CRF
            ('pos_CRF', Pipeline([
                  ('selector', ItemSelector(key='pos_CRF')),
                  ('tfidf', TfidfVectorizer()),
              ])),

            # Pipeline for pulling features from post's PoS tagging with HMM
            # ('pos_HMM', Pipeline([
            #     ('selector', ItemSelector(key='pos_HMM')),
            #     ('tfidf', TfidfVectorizer()),
            # ])),
            #
            # Pipeline for pulling features from post's parse tree
            # Uses dependency tags as features - simple counter
            # ('parse_tree_tags', Pipeline([
            #      ('selector', ItemSelector(key='parse_trees')),
            #      ('vect', CountVectorizer()),
            # ])),

            # Pipeline for pulling stats from post's parse tree
            ('parse_tree_stats', Pipeline([
                ('selector', ItemSelector(key='tree_stats')),
                ('stats', TreeStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

            # Pipeline for actions
            ('actions', Pipeline([
                ('selector', ItemSelector(key='action')),
                ('vect', CountVectorizer()),  # Count Vectorizer for now
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={
             # 'pos': 1.0,
             'dialogue_bow': 1.0,
             #'dialogue_stats': 1.0,
             'actions': 0.2,
             'word2vec': 0.2,
             'pos_CRF': 0.2,
             # 'pos_HMM': 1.0,
             # 'parse_tree_tags': 1.0,
             'parse_tree_stats': 0.2,
        },
    )),

    # Classifier
    # ('svc', SVC(kernel='linear')),
    #('svc', LinearSVC()),
    # ('clf', MLPClassifier()),
    # ('clf', MultinomialNB()),
    # ('clf', DecisionTreeClassifier()),
    # ('clf', RandomForestClassifier()),
    ('clf', LogisticRegression()),
])


# ----------------------------------------------------------------
# ----------------------------- MAIN -----------------------------
# Perform the preprocessing of the actions column of the dataframe.
# Extracting the list of the actions of the characters from text containing the descriptions of the scene
train_df.update(treat_action(train_df))
test_df.update(treat_action(test_df))

# Spliting the dataset into: x - dialogue and action, y - character
X_train = train_df[['dialogue', 'action']]
y_train = train_df.character
X_test = test_df[['dialogue', 'action']]
y_test = test_df.character

# Performing the cross-validation on the training dataset
chosen_classifier = crossValidate(train_df,10)

# Results with the chosen classifier
# Predict lables for the training dataset
y_pred = chosen_classifier.predict(X_train)
# Save model in a file
save_obj(chosen_classifier,"trained_model")
# Calculate the train accuracy
print('Train accuracy %s' % accuracy_score(y_train, y_pred))
# Find the precision, recall adn fscore
cv = precision_recall_fscore_support(y_train, y_pred, average='weighted')
print(cv)
# Predict lables for the test dataset
y_pred = chosen_classifier.predict(X_test)
# Calculate the test accuracy
print('Test accuracy %s' % accuracy_score(y_test, y_pred))
# Find the precision, recall, fscore
cv = precision_recall_fscore_support(y_test, y_pred, average='weighted')
