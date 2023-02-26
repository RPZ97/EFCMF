import os
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy import sparse as sp
from tqdm import tqdm


def normalizeAdj(mx:sp.coo_matrix):
    """Row-normalize sparse matrix"""
    mxNew = mx + mx.T.multiply(mx.T > mx) - mx.multiply(mx.T > mx)
    mxNew:sp.coo_matrix = mxNew + sp.eye(mxNew.shape[0])
    return mxNew

def loadCustomTextGraph(pathDict:Dict[str,str]):
    kgNodeDict:Dict[str,Dict[str,str]] = {}
    categoryList,U,V = [],[],[]
    for path in [i for i in pathDict.values()]:
        for category in os.listdir(path):
            if category not in categoryList:
                categoryList.append(category)
    categoryList.sort()
    documentNumber = 0
    for setName,path in pathDict.items():
        for root,_,fileNameList in tqdm(os.walk(path)):
            fileCategory = os.path.basename(root)
            for fileName in fileNameList:
                docName = f"doc-{documentNumber}"
                kgNodeDict.update({f"{docName}":{
                    "index":len(kgNodeDict.keys()),
                    "set":setName,
                    "category":fileCategory}})
                documentNumber +=1
                filePath = os.path.join(root,fileName)
                with open(filePath,"r") as file:
                    data = file.read().splitlines()
                    for i,sentence in enumerate(data):
                        wordList = sentence.split(" ")
                        sentenceName = f"doc-{documentNumber}-{i}"
                        kgNodeDict.update({sentenceName:{"index":len(kgNodeDict.keys())}})
                        U.append(kgNodeDict[docName]["index"])
                        V.append(kgNodeDict[sentenceName]["index"])    
                        for k in range(len(wordList)):
                            word = wordList[k]
                            if word not in kgNodeDict.keys():
                                kgNodeDict.update({word:{"index":len(kgNodeDict.keys())}})
                            if k>=1:
                                wordPref = wordList[k-1]
                                wordPair = [wordPref,word]
                                wordPairString = "-".join(wordPair)
                                if wordPairString not in kgNodeDict.keys():
                                    kgNodeDict.update({wordPairString:{"index":len(kgNodeDict.keys())}})
                                    U.append(kgNodeDict[wordPairString]["index"])
                                    V.append(kgNodeDict[word]["index"])
                                    U.append(kgNodeDict[wordPairString]["index"])
                                    V.append(kgNodeDict[wordPref]["index"])
                                U.append(kgNodeDict[sentenceName]["index"])
                                # U.append(kgNodeDict[docName]["index"])
                                V.append(kgNodeDict[wordPairString]["index"])
                            # U.append(kgNodeDict[docName]["index"])
                            # V.append(kgNodeDict[word]["index"])
    spGraph = sp.coo_matrix(([1 for i in range(len(U+V))],(U+V,V+U)),shape=(len(kgNodeDict.keys()),len(kgNodeDict.keys())))
    returnData = {"graph":normalizeAdj(spGraph)}
    returnData.update({"kgNodeDict":kgNodeDict})
    for setName in pathDict.keys():
        returnData.update(
            {
                f"{setName}Index":[value["index"] for key,value in kgNodeDict.items() if ("set" in value.keys() and value["set"]==setName)],
                f"{setName}LabelNameList":[value["category"] for key,value in kgNodeDict.items() if ("set" in value.keys() and value["set"]==setName)],
            })
    return returnData

def loadTextGraph(pathDict:Dict[str,str],window_size=20,missingRatio=0):
        # constructing all windows
    kgNodeDict:Dict[str,Dict[str,str]] = {}
    doc_list:List[str] = []
    word_id_map:Dict[str,int] = {}
    vocab:List[str] = []
    word_doc_freq:Dict[str,int] = {}
    documentNumber = 0
    for setName,path in pathDict.items():
        for root,_,fileNameList in tqdm(os.walk(path)):
            fileCategory = os.path.basename(root)
            for fileName in fileNameList:
                docName = f"doc-{documentNumber}"
                kgNodeDict.update({f"{docName}":{
                    "index":len(kgNodeDict.keys()),
                    "set":setName,
                    "category":fileCategory}})
                documentNumber +=1
                filePath = os.path.join(root,fileName)
                with open(filePath,"r") as file:
                    data = file.read()
                    doc_list.append(data)
                    data = data.splitlines()
                    for i,sentence in enumerate(data):
                        wordList = sentence.split(" ")
                        for k in range(len(wordList)):
                            word = wordList[k]
                            if word not in kgNodeDict.keys():
                                kgNodeDict.update({word:{"index":len(kgNodeDict.keys())}})
                                word_id_map.update({word:kgNodeDict[word]["index"]})
                                word_doc_freq.update({word:0})
                                vocab.append(word)
                            word_doc_freq[word] +=1
    # --------------------------------------------
    windows:List[List[str]] = []
    for doc_words in doc_list:
        words = doc_words.split()
        doc_length = len(words)
        if doc_length <= window_size:
            windows.append(words)
        else:
            for i in range(doc_length - window_size + 1):
                window = words[i: i + window_size]
                windows.append(window)
    # constructing all single word frequency
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        for word in window:
            if word not in appeared:
                word_window_freq[word] += 1
                appeared.add(word)
    # constructing word pair count frequency
    word_pair_count = defaultdict(int)
    for step, window in enumerate(tqdm(windows)) :
        for i in range(1, len(window)):
            for j in range(i):
                word_i = window[i]
                word_j = window[j]
                # word_i_id = word_id_map[word_i]
                # word_j_id = word_id_map[word_j]
                # if word_i_id == word_j_id:
                #     continue
                if word_i == word_j:
                    continue
                # word_pair_count[(word_i_id, word_j_id)] += 1
                # word_pair_count[(word_j_id, word_i_id)] += 1
                word_pair_count[(word_i, word_j)] += 1
                word_pair_count[(word_j, word_i)] += 1
    row = []
    col = []
    weight = []

    # pmi as weights
    num_docs = len(doc_list)
    num_window = len(windows)
    for step,(word_id_pair, count)in enumerate(tqdm(word_pair_count.items())) :
        i, j = word_id_pair[0], word_id_pair[1]
        # word_freq_i = word_window_freq[vocab[i]]
        # word_freq_j = word_window_freq[vocab[j]]
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        pmi = np.log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        # row.append(num_docs + i)
        # col.append(num_docs + j)
        # row.append(num_docs + kgNodeDict[i]["index"])
        # col.append(num_docs + kgNodeDict[j]["index"])
        row.append(kgNodeDict[i]["index"])
        col.append(kgNodeDict[j]["index"])
        weight.append(pmi)

    # frequency of document word pair
    doc_word_freq = defaultdict(int)
    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = (i, word_id)
            doc_word_freq[doc_word_str] += 1

    for i, doc_words in enumerate(doc_list):
        words = doc_words.split()
        doc_word_set = set()
        if missingRatio>0:
            random.seed(i)
            random.shuffle(words)
        for word in words[:int((1-missingRatio)*len(words))]:
            if word in doc_word_set:
                continue
            word_id = word_id_map[word]
            freq = doc_word_freq[(i, word_id)]
            # row.append(i)
            row.append(kgNodeDict[f"doc-{i}"]["index"])
            # col.append(num_docs + word_id)
            col.append(kgNodeDict[word]["index"])
            # idf = np.log(1.0 * num_docs /word_doc_freq[vocab[word_id]])
            idf = np.log(1.0 * num_docs /word_doc_freq[word])
            weight.append(freq * idf)
            doc_word_set.add(word)

    number_nodes = num_docs + len(vocab)
    # adj_mat = sp.csr_matrix((weight, (row, col)), shape=(number_nodes, number_nodes))
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(len(kgNodeDict.keys()), len(kgNodeDict.keys())))
    adj = adj_mat + adj_mat.T.multiply(adj_mat.T > adj_mat) - adj_mat.multiply(adj_mat.T > adj_mat)
    returnData = {"graph":adj}
    for setName in pathDict.keys():
        returnData.update(
            {
                f"{setName}Index":[value["index"] for key,value in kgNodeDict.items() if ("set" in value.keys() and value["set"]==setName)],
                f"{setName}LabelNameList":[value["category"] for key,value in kgNodeDict.items() if ("set" in value.keys() and value["set"]==setName)],
            })
    return returnData