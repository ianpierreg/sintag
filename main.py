# -*- coding: utf-8 -*-
import nltk
from cogroo_interface import Cogroo
cogroo = Cogroo.Instance()
import operator
probability_matrix = {
    "BNP": {"BNP": 0.0, "INP": 0.0, "ENP": 0.0, "BVP": 0.0, "IVP": 0.0, "EVP": 0.0, "COUNT": 0},
    "INP": {"BNP": 0.0, "INP": 0.0, "ENP": 0.0, "BVP": 0.0, "IVP": 0.0, "EVP": 0.0, "COUNT": 0},
    "ENP": {"BNP": 0.0, "INP": 0.0, "ENP": 0.0, "BVP": 0.0, "IVP": 0.0, "EVP": 0.0, "COUNT": 0},
    "BVP": {"BNP": 0.0, "INP": 0.0, "ENP": 0.0, "BVP": 0.0, "IVP": 0.0, "EVP": 0.0, "COUNT": 0},
    "IVP": {"BNP": 0.0, "INP": 0.0, "ENP": 0.0, "BVP": 0.0, "IVP": 0.0, "EVP": 0.0, "COUNT": 0},
    "EVP": {"BNP": 0.0, "INP": 0.0, "ENP": 0.0, "BVP": 0.0, "IVP": 0.0, "EVP": 0.0, "COUNT": 0}
}


def pre_process():
    global text
    
    if 'ao' in text:
        text.replace('ao', 'a o')
    if 'à' in text:
        text.replace('à', 'a a')
    if 'da' in text:
        text.replace('da', 'de a')
    if 'do' in text:
        text.replace('do', 'de o')
    if 'na' in text:
        text.replace('na', 'em a')
    if 'no' in text:
        text.replace('no', 'em o')
    if 'dele' in text:
        text.replace('dele', 'de ele')
    if 'dela' in text:
        text.replace('dela', 'de ela')
    if 'pela' in text:
        text.replace('pela', 'por a')
    if 'pelo' in text:
        text.replace('pelo', 'por o')
    if 'nele' in text:
        text.replace('nele', 'em ele')
    if 'nela' in text:
        text.replace('nela', 'em ela')
    if 'dali' in text:
        text.replace('dali', 'de ali')
    if 'àquele' in text:
        text.replace('àquele', 'a aquele')
    if 'àquela' in text:
        text.replace('àquela', 'a aquela')
    if 'àquilo' in text:
        text.replace('àquilo', 'a aquilo'))


def get_sentences():
    global conj_sentences, text
    
    conj_sentences = text.split(".")
    if '' in conj_sentences:
        conj_sentences.remove('')
    

def read_files(file_name):
    global text
    
    with open(file_name) as file:
        text = file.read()


def get_morphology():
    global cogroo, pos_taggers, conj_sentences
    
    pos_taggers = []
    for term in conj_sentences:
        tagger = []        
        sentence = cogroo.analyze(term)
        for pos_tag in sentence.sentences[0].tokens:
            tagger.append(pos_tag.pos)
        pos_taggers.append(tagger)


def analyze_sentences():
    read_files('entrada.txt')
    pre_process() # realiza a separação de palavras compostas
    get_sentences()
    get_morphology()


def count_sintag(sentence):
    words_sentence = sentence.split()
    for word in words_sentence:
        splitted_word = word.split('/')
        probability_matrix[splitted_word[1][:3]]["COUNT"] += 1


def get_probability(sentence):
    last_splitted = None
    words_sentence = sentence.split()
    for word in words_sentence:
        splitted_word = word.split('/')
        sintag_splitted = splitted_word[1][:3]
        if (last_splitted):
            probability_matrix[last_splitted][sintag_splitted] += (1 / probability_matrix[last_splitted]["COUNT"])
        last_splitted = sintag_splitted


def define_first_sintag(word):
    if "VB" in nltk.pos_tag([word])[0][1]:
        return "/BVP"
    return "/BNP"


def test_sentences(sentences):
    senteces_no_sintag = []
    senteces_new_sintag = []
    for sentence in sentences:
        splitted_words = sentence.split()

        sentence_without_sintag = ""
        for splitted_word in splitted_words:
            sentence_without_sintag += splitted_word.split('/')[0]+" "
        senteces_no_sintag.append(sentence_without_sintag)
        senteces_new_sintag.append(' '.join(get_next_sentence_sintag(sentence_without_sintag)))
        
    test_x_train(sentences, senteces_new_sintag)


def get_next_sentence_sintag(sentence):
    words_sentence = sentence.split()
    last_word = words_sentence[0] + define_first_sintag(words_sentence[0])
    words = [last_word]

    for word in words_sentence[1:]:
        splitted_word = last_word.split('/')
        sintag_splitted = splitted_word[1][:3]
        probability_matrix_copy = probability_matrix[sintag_splitted].copy()
        del(probability_matrix_copy["COUNT"])
        word_sintag = sorted(probability_matrix_copy.items(), key=operator.itemgetter(1), reverse=True)
        word = word+"/"+word_sintag[0][0]
        words.append(word)
        last_word = word
    return words


def test_x_train(sentences_old_sintag, sentences_new_sintag):
    general_count = 0
    match_sintag = 0
    match_phrase = 0

    for sentence1, sentence2 in zip(sentences_old_sintag, sentences_new_sintag):

        words_sentence_1 = sentence1.split()
        words_sentence_2 = sentence2.split()
        equal_phrase = True
        for word_1, word_2 in zip(words_sentence_1, words_sentence_2):
            general_count += 1
            if word_1.split('/')[1][:3] == word_2.split('/')[1][:3]:
                match_sintag += 1
            else:
                equal_phrase = False
        if equal_phrase:
            match_phrase += 1
    token_similarity = (match_sintag / general_count) * 100
    sentence_similarity = (match_phrase / len(sentences_old_sintag)) * 100
    print("Match de tokens/Total de tokens %s/%s" % (match_sintag, general_count))
    print("Simlaridade por tokens: %.2f%%\n\n" % token_similarity)


    print("Match de frases/Total de frases %s/%s" % (match_phrase, len(sentences_old_sintag)))
    print("Similaridade por frase:%.2f%%" % sentence_similarity)


# sentence = 'Daniela/BNP Claro/ENP é/BVP professora/BNP de/INP a/INP UFBA/ENP'
# sentence2 = 'Daniela Claro é professora de a UFBA'

with open('training.txt') as f:
    train_set = f.readlines()

for sentence in train_set:
    count_sintag(sentence)

for sentence in train_set:
    get_probability(sentence)

with open('test.txt') as f:
    test_set = f.readlines()
test_sentences(test_set)



