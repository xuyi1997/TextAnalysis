
import re
import csv
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

import opennre
opennre_model = opennre.get_model('wiki80_cnn_softmax')

with open('stopwords_en.txt', 'r', encoding='utf-8' ) as f:
    stopwords = [line.strip() for line in f if line.strip()]


with open('speech_NeuronLecture.txt', 'r', encoding='utf-8' ) as f:
    doc = f.read()

entity_deps = {'nsubj', 'nsubjpass', 'dobj', 'compound', 'appos', 'attr'}
stopwords += ["'s", "'m", "'re", "'ve", "'ll", "n't", "'d", "it", "you", "I", "they", "he", "she", "we", "they", "this", "that"] 
min_length = 2  # 设置最小词长

triples = []
doc_nlp = nlp(doc)
for sent in doc_nlp.sents:
    text = sent.text
    entities = []
    ents = list(sent.ents)
    for token in sent:
        is_entity = any(token.i >= ent.start and token.i < ent.end for ent in ents)
        if (is_entity or token.dep_ in entity_deps)  and token.text not in stopwords and len(token.text) >= min_length and doc.count(token.text) > 1:
            entities.append(token.text)
    if entities:
         for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                subj, obj = entities[i], entities[j]
                if subj == obj:
                    continue
                pos1 = (text.index(subj), text.index(subj) + len(subj))
                pos2 = (text.index(obj), text.index(obj) + len(obj))
                rel, score = opennre_model.infer({'text': text, 'h': {'pos': pos1}, 't': {'pos': pos2}})
                if score > 0.5:
                    triples.append([subj, rel, obj])
                    print(f"Found triple: {subj} - {rel} ({score:.2f}) - {obj}")


print(triples)
fh = open("opennre_tripples.txt", "w", encoding="utf-8")
print(triples, file=fh)
fh.close()

