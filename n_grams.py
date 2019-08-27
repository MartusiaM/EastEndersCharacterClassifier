import re                # Regex
import pickle            # Loading and saving objects with pickle
import collections
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import ngrams

# ------------------------------------------
#  PICKLE FUNCTIONS FOR SAVING AND LOADING


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------
# ---------------- HELPER FUNCTIONS -----------------

def clean_dialogue(dialogue):
    return re.sub('\s+', ' ', dialogue.replace("\n", " ")).strip()


def decontracted(phrase):
    print(phrase)
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

# ---------------------------------------------------
# ----------------------- MAIN ----------------------

n = 3

train_dictionary = load_obj("train_dictionary")
test_dictionary = load_obj("test_dictionary")
stopwords = set(stopwords.words('english'))
dialogue = train_dictionary["TANYA"]
for key in train_dictionary:
    dialogue = train_dictionary[key]

    all_ngrams = []

    for utt in dialogue:
        sentences = sent_tokenize(utt)
        for sentence in sentences:
            dcsentence = sentence.translate(str.maketrans('', '', string.punctuation))
            # tokens = dcsentence.split()
            text_ngrams = ngrams(sentence.split(), n)
            # text_ngrams = nltk.trigrams(tokens)
            all_ngrams += text_ngrams
    collections.Counter(all_ngrams).most_common()[0]

    for tup in all_ngrams:
        if (tup[0] in stopwords) and (tup[1] in stopwords) and (tup[2] in stopwords):
            all_ngrams.remove(tup)
    print("nmost common ngrams for: ", " are", sep=key)
    nmostcommon = []
    for i in range(5):
        nmostcommon += [collections.Counter(all_ngrams).most_common()[i]]
    print(nmostcommon)
