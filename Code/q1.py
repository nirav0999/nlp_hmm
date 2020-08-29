#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import csv
import pickle
import math
import pandas as pd
def appendToCSV(filepath,row):
	with open(filepath,"a",buffering = 1) as csvfile:
		writer = csv.writer(csvfile)
		wrtiter.writerow(row)

def openCSVfile(filepath,delim = ","):
	with open(filepath,"r") as csvfile:
		rows =  csv.reader(csvfile,delimiter = delim)
		return list(rows)
    
def dumpJsonFile(filepath,dictionary):
	print("Dumping a dictionary to filepath",filepath,"...............")
	with open(filepath,"w+") as jsonFile:
		json.dump(dictionary,jsonFile,indent=4,sort_keys =True)
	print("Dumped Successfully")

def LoadJsonFile(filepath):
	print("Loading a dictionary to filepath",filepath,"...............")
	dictionary = {}
	with open(filepath) as jsonFile:
		dictionary = json.load(jsonFile)
	print("Loaded Successfully")
	return dictionary

def loadPickleFile(filepath):
	print("Loading the pickle file from",filepath,"...........")
	pickle_in = open(filepath,"rb")
	example_dict = pickle.load(pickle_in)
	print("Loaded the pickle File")
	return example_dict

def dumpPickleFile(data,filepath):
	pickle_out = open(filepath,"wb")
	print("Dumping the Pickle file into ",filepath,"...........")
	pickle.dump(data, pickle_out)
	print("Dumped the pickle File")
	pickle_out.close() 


# In[57]:


def insertTags(dictionary,ftag,stag = None,gram = 2):
    """
    Input - 
    Output - 
    Returns -
    """
    if gram == 2:
        if ftag in dictionary:
            if stag in dictionary[ftag]:
                dictionary[ftag][stag] += 1
            else:
                dictionary[ftag][stag] = 1
        else:
            dictionary[ftag] = {stag : 1}
        return dictionary
    elif gram == 1:
        if ftag in dictionary:
            dictionary[ftag] += 1
        else:
            dictionary[ftag] = 1
    return dictionary


def MakeMatrices(rows):
    bigramTagDict = {}
    unigramTagDict = {}
    unigramWordDict = {}
    wordTagDict = {}
    endingIndex = -1
    for row_no in range(0,len(rows) - 1):
        currentRow = rows[row_no]
        nextRow = rows[row_no + 1]
        #print("Current Row = ",currentRow)
        #print("Next Row = ",nextRow)
        if currentRow[0] == nextRow[0] == "<startTag>":
            endingIndex = row_no
            break
        if currentRow[0] == "<startTag>":
            
            bigramTagDict = insertTags(bigramTagDict,"<startTag>",nextRow[1],2)
            unigramTagDict = insertTags(unigramTagDict,"<startTag>",gram = 1)
            unigramWordDict = insertTags(unigramWordDict,"<startTag>",gram = 1)
        elif currentRow[0] == "<endTag>":
            unigramTagDict = insertTags(unigramTagDict,"<endTag>",gram = 1)
            unigramWordDict = insertTags(unigramWordDict,"<endTag>",gram = 1)
            continue
        else:
            bigramTagDict = insertTags(bigramTagDict,currentRow[1],nextRow[1],2)
            unigramTagDict = insertTags(unigramTagDict,currentRow[1],gram = 1)
            unigramWordDict = insertTags(unigramWordDict,currentRow[0],gram = 1)
            wordTagDict = insertTags(wordTagDict,currentRow[0],currentRow[1],2)
    #print(bigramTagDict)
    #print("--------------------------------------")
    #print(unigramTagDict)
    #print("--------------------------------------")
    #print(wordTagDict)
    rows = rows[:endingIndex]
    return rows,unigramWordDict,unigramTagDict,bigramTagDict,wordTagDict

def readTrainingSet(filepath = "../Dataset/Training set_HMM.txt"):
    rows = openCSVfile(filepath,"\t")
    for row_no in range(len(rows)):
        row = rows[row_no]
        nfitems = len(row)
        if nfitems == 0:
            rows[row_no] = ["<startTag>","<startTag>"]
        else:
            if rows[row_no][0] == ".":
                rows[row_no] = ["<endTag>","<endTag>"]
    rows.insert(0,["<startTag>","<startTag>"])
    return rows

def readTestingSet(filepath = "../Dataset/Test_HMM.txt"):
    rows = []
    with open(filepath) as file:
        r1 = file.readlines()
    for r in r1:
        r = r.strip("\n")
        rows.append(r)
    for row_no in range(len(rows)):
        row = rows[row_no]
        nfitems = len(row)
        if nfitems == 0:
            rows[row_no] = "<startTag>"
        else:
            if rows[row_no][0] == ".":
                rows[row_no] = "<endTag>"
    rows.insert(0,"<startTag>")
    return rows   

def calcTransitionProbs(unigramTagDict,bigramTagDict):
    bigramTagProbDict = {}
    for fword in bigramTagDict.keys():
        bigramTagProbDict[fword] = {} 
        for sword in bigramTagDict[fword].keys():
            prob = float(bigramTagDict[fword][sword])/float(unigramTagDict[fword])
            logprob = math.log(prob,10)
            bigramTagProbDict[fword][sword] = logprob
    return bigramTagProbDict

            
def numberOfWords(unigramTagDict):
    totalWords = 0
    falseWords = ["<endTag>","<startTag>"]
    for fword in unigramTagDict.keys():
        if fword not in falseWords:
            totalWords += unigramTagDict[fword]
    return totalWords,(len(unigramTagDict) - 2)
    
def calcWordTagProbs(unigramTagDict):
    totalWords,uniqueWords = numberOfWords(unigramTagDict)
    unigramTagProbDict = {}
    for word in unigramTagDict.keys():
        prob = float(unigramTagDict[word])/float(totalWords)
        logprob = math.log(prob,10)
        unigramTagProbDict[word] = logprob
    return unigramTagProbDict
            
def modifyOutput(filename = "OUTPUT.txt"):
    finallines = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip("\n")
            line1 = line.split("\t")
            #print(line1)
            if line1[0] == "<startTag>":
                finallines.append([])
            elif line1[0] == "<endTag>":
                finallines.append([".","."])
            else:
                finallines.append(line1)
    df = pd.DataFrame(finallines)
    print("FInal Results = ")
    print(df)
    df.to_csv("RESULTS.txt",sep = "\t")
                
            


# In[49]:


rows = readTrainingSet()
rows1 = readTestingSet()
#print(rows1)


# In[44]:


rows,unigramWordDict,unigramTagDict,bigramTagDict,wordTagDict = MakeMatrices(rows)


# In[45]:


#tokens = ["<startTag>","the","distance","does","not","matter","<endTag>"]
tokens = rows1
bigramTagProbDict = calcTransitionProbs(unigramTagDict,bigramTagDict)
unigramTagProbDict = calcWordTagProbs(unigramTagDict)


# In[58]:


def getUniqueTokens(unigramTagDict):
    falseTags = []
    uniqueTags = []
    for key in unigramTagDict.keys():
        if key not in falseTags:
            uniqueTags.append(key)
    uniqueTags.sort()
    return uniqueTags

def calcProbablities(tokens,unigramWordDict,unigramTagDict,bigramTagProbDict,bigramTagDict,wordTagDict,k = 1):
    falseTags = ["<endTag>","<startTag>"]
    uniqueTags = getUniqueTokens(unigramTagDict)
    nftags = len(uniqueTags)
    nftokens = len(tokens)
    colHeadings = {}
    for i in range(len(tokens)):
        colHeadings[i] = tokens[i]
    probMatrice = []
    for i in range(nftags):
        newM = []
        for i in range(nftokens):
            newM.append(0)
        probMatrice.append(newM)
    prevWord = "<startTag>"
    for token_no in range(1,len(tokens)):
        currWord = tokens[token_no]
        for tag_no_old in range(len(uniqueTags)):
            prevTag = uniqueTags[tag_no_old]
            for tag_no in range(len(uniqueTags)):
                currTag = uniqueTags[tag_no]
                p = 0
                p1 = 0
                if prevTag in bigramTagDict:
                    if currTag not in bigramTagDict[prevTag]:
                        p = 0
                    else:
                        p =  10**bigramTagProbDict[prevTag][currTag]
                else:
                    p = 0
                TagCount = unigramTagDict[currTag]
                if currWord not in wordTagDict:
                    p1 = float(k)/float(TagCount + k*nftags)
                else:
                    if currTag not in wordTagDict[currWord]:
                        p1 = 0
                    else:
                        p1 = float(wordTagDict[currWord][currTag] + k)/float(TagCount + k*nftags)
                if token_no - 1 == 0:
                    newProb = p * p1
                else:
                    newProb = p * p1 * probMatrice[tag_no_old][token_no - 1]
                if newProb > probMatrice[tag_no][token_no]:
                    probMatrice[tag_no][token_no] = newProb
                    #maxMatrice[tag_no][token_no] = prevTag 
    rowDict = {}
    for i in range(nftags):
        rowDict[i] = uniqueTags[i]
    df1 = pd.DataFrame(probMatrice)
    df1.rename(index = rowDict,inplace = True)
    df2 = df1.idxmax(axis = 0) 
    df2.rename(index = colHeadings,inplace = True)
    df2.to_csv("OUTPUT.txt",sep = "\t")
    modifyOutput()
    #print(df2)

calcProbablities(tokens,unigramWordDict,unigramTagDict,bigramTagProbDict,bigramTagDict,wordTagDict,k = 1)


# In[ ]:




