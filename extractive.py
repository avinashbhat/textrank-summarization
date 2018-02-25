import numpy as np
import re
from operator import itemgetter

text = """Mohandas Karamchand Gandhi was an Indian activist who was the leader of the 
        Indian independence movement against British rule. Employing nonviolent civil 
        disobedience, Gandhi led India to independence and inspired movements for civil 
        rights and freedom across the world. In India, he is also called 
        Bapu and Gandhi ji. He is unofficially called the Father of the Nation. Born and 
        raised in a Hindu merchant caste family in coastal Gujarat, India, and trained
        in law at the Inner Temple, London, Gandhi first employed nonviolent civil 
        disobedience as an expatriate lawyer in South Africa, in the resident Indian 
        community's struggle for civil rights. After his return to India in 1915, he 
        set about organising peasants, farmers, and urban labourers to protest against 
        excessive land-tax and discrimination. Assuming leadership of the Indian 
        National Congress in 1921, Gandhi led nationwide campaigns for various social 
        causes and for achieving Swaraj or self-rule. Gandhi famously led Indians in 
        challenging the British-imposed salt tax with the 400 km Dandi Salt 
        March in 1930, and later in calling for the British to Quit India in 1942. He 
        was imprisoned for many years, upon many occasions, in both South Africa and 
        India. He lived modestly in a self-sufficient residential community and wore 
        the traditional Indian dhoti and shawl, woven with yarn hand-spun on a charkha. 
        He ate simple vegetarian food, and also undertook long fasts as a means of 
        both self-purification and political protest. Gandhi's vision of an independent 
        India based on religious pluralism, however, was challenged in the early 1940s 
        by a new Muslim nationalism which was demanding a separate Muslim homeland 
        carved out of India. Eventually, in August 1947, Britain granted independence, 
        but the British Indian Empire was partitioned into two dominions, a 
        Hindu-majority India and Muslim-majority Pakistan. As many displaced Hindus, 
        Muslims, and Sikhs made their way to their new lands, religious violence broke 
        out, especially in the Punjab and Bengal. Eschewing the official celebration of 
        independence in Delhi, Gandhi visited the affected areas, attempting to provide 
        solace. In the months following, he undertook several fasts unto death to stop 
        religious violence. The last of these, undertaken on 12 January 1948 when he was 
        78, also had the indirect goal of pressuring India to pay out some cash 
        assets owed to Pakistan. Some Indians thought Gandhi was too accommodating. 
        Among them was Nathuram Godse, a Hindu nationalist, who assassinated Gandhi on 
        30 January 1948 by firing three bullets into his chest. Captured along with 
        many of his co-conspirators and collaborators, Godse and his co-conspirator 
        Narayan Apte were tried, convicted and executed while many of their other 
        accomplices were given prison sentences. Gandhi's birthday, 2 October, is 
        commemorated in India as Gandhi Jayanti, a national holiday, and worldwide as 
        the International Day of Nonviolence."""

# Cosine Similarity = cos(theta) = (A . B) / (||A|| ||B||)
def cosineSimilarity(vector1, vector2):
    dotProduct = np.dot(vector1, vector2)
    normV1 = np.linalg.norm(vector1)
    normV2 = np.linalg.norm(vector2)
    return dotProduct / (normV1 * normV2)

# A function to calculate the similarity between two sentences
def sentenceSimilarity(sentence1, sentence2, stopwords=None):
    if stopwords is None:
        stopwords = []

    # Reduce all words to lower case
    sentence1 = [word.lower() for word in sentence1]
    sentence2 = [word.lower() for word in sentence2]
    # Put all words into a set, so that only one occurance is present
    allWords = list(set(sentence1 + sentence2))
 
    vector1 = [0] * len(allWords)
    vector2 = [0] * len(allWords)
 
    # build the vector for the first sentence
    # Vector is nothing but setting the index of the word
    # For example, if sentence1 is 'My name is Avinash'
    # And sentence2 is 'I like Chocolates'
    # Vector1 = [1,1,1,1,0,0,0], Vector2 = [0,0,0,0,1,1,1] 
    # where 1s correspond to the words present
    for word in sentence1:
        if word in stopwords:
            continue
        vector1[allWords.index(word)] += 1
 
    # build the vector for the second sentence
    for word in sentence2:
        if word in stopwords:
            continue
        vector2[allWords.index(word)] += 1
 
    return cosineSimilarity(vector1, vector2)

# A function to clean the text of whitespaces
# Returns list of lists
def processText(text):
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub(' +', ' ', text)
    sentenceList = list()
    wordList = list()
    tempList = text.strip().split(".")
    for sentence in tempList:
        wordList = sentence.strip().split(" ")
        if len(wordList) > 1:
            sentenceList.append(wordList)
    return sentenceList, len(sentenceList)

sentences, length = processText(text)    
#print(sentences)
#print(length)
 
# get the english list of stopwords
#stop_words = stopwords.words('english')
 
 # A function to compute the similarity matrix
def buildSimilarityMatrix(sentences, stopwords=None):
    # Create an empty similarity matrix
    S = np.zeros((len(sentences), len(sentences)))

    for index1 in range(len(sentences)):
        for index2 in range(len(sentences)):
            # To eliminate calculating similarity of same sentences
            if index1 == index2:
                continue
            # If sentences are not same calculate the similarity
            S[index1][index2] = sentenceSimilarity(sentences[index1], sentences[index2], stopwords)
 
    # normalize the matrix row-wise
    for index in range(len(S)):
        S[index] /= S[index].sum()
 
    return S
 
S = buildSimilarityMatrix(sentences)    
#print(S)

# Function to rank the sentences using textrank algo
# Stop the algorithm when the difference between 2 consecutive iterations is smaller or equal to eps
# With a probability of 1-d, simply pick a sentence at random as the next valid sentence
def ranking(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        newP = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((newP - P).sum())
        if delta <= eps:
            return newP
        P = newP

# Driver function for the extractive code
def textrank(sentences, linesinSummary=10, stopwords=None):
    S = buildSimilarityMatrix(sentences, stopwords) 
    sentenceRanks = ranking(S)
 
    # Sort the sentence ranks
    rankedSentenceIndexes = [item[0] for item in sorted(enumerate(sentenceRanks), key=lambda item: -item[1])]
    selectedSentences = sorted(rankedSentenceIndexes[:linesinSummary])
    summary = itemgetter(*selectedSentences)(sentences)
    return summary

# Create summary
for index, sentence in enumerate(textrank(sentences)):
    print("%s." % (' '.join(sentence)))