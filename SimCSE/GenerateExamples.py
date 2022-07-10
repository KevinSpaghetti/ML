import pandas as pd
from nltk.corpus import wordnet2021 as wd
from sklearn.model_selection import train_test_split
import nltk

from simcse import SimCSE
import random
import numpy as np

'''
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('wordnet2021')
'''

seed = 42
random.seed(seed)
np.random.seed(seed)

df = pd.read_csv('./datasets/combined.csv', sep=",")

train, test = train_test_split(df, test_size=0.8)

model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")


def get_synonyms(words):
    synonyms = {word: [] for word in words}

    for word in words:
        if len(word) > 2:
            synsets = wd.synsets(word)
            if synsets:
                synset = synsets[0]
                for lemma in synset.lemmas():
                    synonyms[word].append(lemma.name())

    return synonyms


def get_meaning(words):
    definitions = {word: [] for word in words}

    for word in words:
        if len(word) > 2:
            synset = wd.synsets(word)
            if synset:
                definitions[word].append(synset[0].definition())

    return definitions


ades = train['ade']
meddras = train['meddra']
positives = []
negatives = []

# Calcoliamo gli encoding che ci serviranno in seguito
train['text_embedding'] = model.encode(train['text'].tolist())
train['ade_embedding'] = model.encode(train['ade'].tolist())
train['meddra_embedding'] = model.encode(train['meddra'].tolist())

# Calcolo dei positives e dei negatives
for index, (text, ade, meddra, _, text_embedding, _, _) in train.iterrows():
    same_ade_meddra_pair = df.loc[(df['ade'] == ade) & (df['meddra'] == meddra)]
    # Se non ci sono righe con lo stesso paio ade, meddra allora l'esempio positivo è il meddra
    positive: str = same_ade_meddra_pair.sample(n=1)['text'].values[0] if not same_ade_meddra_pair.empty else meddra

    # Opzione 1: L'esempio negativo è semplicemente un testo con meddra diverso
    # I testi che hanno lo stesso meddra e che non possiamo prendere
    texts_with_same_meddra = train.loc[train['meddra'] == meddra]['text'].values
    # Dobbiamo fare così perchè potrebbe esserci un testo con più ade che mappano in diversi meddra per cui rischiamo
    # di prendere come esempio negativo un testo con lo stesso meddra ad esempio
    # testo1 ade1 meddra1
    # testo1 ade2 meddra2
    # In questo caso se volessimo trovare un esempio negativo per meddra2 potremmo prendere la riga 1
    # selezionando il testo1 che ha un meddra diverso (che però contiene anche lo stesso meddra2) potremmo quindi
    # addirittura selezionare lo stesso testo come esempio negativo
    different_meddra_rows = train.loc[~train['text'].isin(texts_with_same_meddra)]
    negative: str = different_meddra_rows.sample(n=1)['text'].values[0]

    # Opzione 2: Usiamo il framework SimCSE per cercare frasi con embedding simili al testo con diverso meddra
    embeddings = different_meddra_rows['text_embedding'].tolist()

    # Troviamo l'indice dell'embedding più vicino

    # Con l'indice prendiamo la riga nel vettore dei testi con meddra diverso
    text_embedding = text_embedding
    different_meddra_texts_embedding = embeddings

    positives.append(positive)
    negatives.append(negative)
