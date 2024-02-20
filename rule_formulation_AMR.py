from easynmt import EasyNMT

from nltk import word_tokenize
from nltk import sent_tokenize
from simalign import SentenceAligner

import json

import re
import penman
import sys

import amrlib
from amrlib.graph_processing.annotator import add_lemmas
from amrlib.alignments.rbw_aligner import RBWAligner

fileName = sys.argv[1]
#%%   
# Find a node corresponding targetWord in the graph:
def getTargetWordNode(segmentTokens, aligner, alignments, target):
    # Get target word in English:
    if target in segmentFrTokens:
        targetIndexFr = segmentTokens.index(target)
        
        targetIndexesEn = [i for i in alignments['mwmf'] if i[0]==targetIndexFr]
        if len(targetIndexesEn) > 0:
            targetIndexEn = targetIndexesEn[0][1]
                
            # Get a full name of the graph node:
            if aligner.alignments[targetIndexEn] != None:
                nodeConcepts = [i for i in re.split(',|\(|\"|\'', str(aligner.alignments[targetIndexEn])) if i.strip() != '']
                return nodeConcepts[0]+' / '+nodeConcepts[2]
            else:
                return 'Error!' # Alignment between target word in French and its English instance not found
        else:
            return 'Error!' # Alignment between target word in French and its English instance not found
    else:
        return 'Error!' # Alignment between target word in French and its English instance not found
    
# Extract a subgraph containing target word with full path (all the node) to it:
def getTargetWordSubGraphFullPath(amrGraph, target):
    stringTmp = [i+' ' for i in re.split('\n', inputGraph[0]) if i[0] !='#']

    stringTmp2 = []
    for s in stringTmp:
        stringTmp2+=[i for i in re.split('(:\w+\s|:\w+-\w+\s)', s) if i.strip() !='']
    
    string = []
    for s in stringTmp2:
        string+=[i for i in re.split('(\(|\))', s) if i.strip() !='']    
    
    openListGlobal = []
    openList = []
    subGraph = ""
    subGraphGlobal = []
    
    flag = False
    stop = False
    for i in range(len(string)):
        if flag:
            if string[i] == '(':
                openList.append('(')
                subGraph+=string[i]
            elif string[i] == ')':
                openList.pop()
                if openList == []:
                    flag = False
                    stop = True
                    subGraph+=')'
                    subGraphGlobal.append(subGraph)
                else:
                    subGraph+=string[i]
            else:
                subGraph+=string[i]
        else:
            if target in string[i].strip():
                flag = True
                subGraph+=string[i]
                openList.append('(')
            else:
                if not stop and string[i] == '(':
                    openListGlobal.append('(')
                    subGraphGlobal.append(string[i])
                elif not stop and string[i] == ')':
                    openListGlobal.pop()
                    while subGraphGlobal[-1] != '(':
                        subGraphGlobal.pop()
                    subGraphGlobal.pop()
                    subGraphGlobal.pop()
                elif not stop:
                    subGraphGlobal.append(string[i])
                    
    for i in openListGlobal:
        if i=='(':
            subGraphGlobal.append(')')
            
    resultGraph = ""
    for i in subGraphGlobal:
        resultGraph+=i
    
    # Fix the formatting:
    return penman.decode(resultGraph)
#%%
# Open dictionary of keywords:
f = open("Summaries/data/"+fileName+'.json')

keywords = json.load(f)

f.close()
#%
# Open collection of segments:
f = open("Summaries/data/"+fileName+"_segments.txt", "r")

data = [] 
segmentText = ""
label = "False"
for line in f:
    try:
        if '>>>' in line:
            # memorize previous segment:
            if segmentText != "":
                data.append((segmentText.strip(),label))
                segmentText = ""
            # determine label of current segment:
            label = line.split('>>>')[1].strip()
        else:
            segmentText += line
                
    except ValueError:
        print('Invalid input:',line)

# Memorize the last segment:
data.append((segmentText.strip(),label))

f.close()
#%%
# Define output file:
f_out = open("Summaries/"+fileName+"_summaries.txt","w")
#%    
# Iterate over segments and keywords:
for segmentId in keywords.keys():
    print('Processing segment #'+segmentId)
    segmentFr = data[int(segmentId)-1][0].replace('\n','')
    
    targetWordList = []
    for keywordType in keywords[segmentId].keys():            
        # Load segment keywords:
        for word in keywords[segmentId][keywordType].keys():
            targetWordList.append((word,keywordType))
        
    if targetWordList != []:
        # Translate segment into English:
        model = EasyNMT('opus-mt')
        segmentEn = model.translate(segmentFr, source_lang='fr', target_lang='en')
        
        # Get an AMR graph:
        stog = amrlib.load_stog_model()
        inputGraph = stog.parse_sents([segmentEn])
        
        # Get tokenized representation of segment in French:
        segmentFrTokens = word_tokenize(segmentFr, language='french')
        
        # Get tokenized representation of segment in English:
        penmanGraph = add_lemmas(inputGraph[0], snt_key='snt')
        aligner = RBWAligner.from_penman_w_json(penmanGraph)
        segmentEnTokens = aligner.lemmas
        
        # Get alignments between original version and translation:
        myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")
        alignments = myaligner.get_word_aligns(segmentFrTokens, segmentEnTokens)
        
        # Output original segment
        f_out.write('>>> '+data[int(segmentId)-1][1]+'\n\n')
        f_out.write(data[int(segmentId)-1][0]+'\n\n')
        f_out.write('>> Résumé :\n\n')
        
        summaryGraph = None
        
        targetNodeList = [] # list of found target nodes
        
        # Iterate over keywords:
        for targetWord,targetWordType in targetWordList:
            print(targetWord,targetWordType)
            # Check if a keyword is a single word:
            if len(targetWord.split()) == 1:
                # Find a node corresponding targetWord in the graph:
                targetNode = getTargetWordNode(segmentFrTokens, aligner, alignments, targetWord)
            else:
                # Use only the first word (current limitation of the model):
                targetNode = getTargetWordNode(segmentFrTokens, aligner, alignments, targetWord.split()[0])
                
            if targetNode != 'Error!':            
                # Check if found targetNode is in the graph:
                errorFlag = False
                if targetNode not in inputGraph[0]:
                    #if targetWord in inputGraph[0]:
                    if targetWord in ''.join(inputGraph[0].split('\n')[1:]):
                        targetNode = targetWord
                    else:
                        errorFlag = True
                
                # Memorise found taget nodes:
                if not errorFlag:
                    targetNodeList.append(targetNode)
                    
        # Check if there are more than 3 found target nodes then use them to generate summaries.
        # Otherwise, the full graph:
        if len(targetNodeList) < 3:
            # Use the full graph:
            summaryGraph = inputGraph[0]
        else:
            # Iterate over target nodes:
            for targetNode in targetNodeList:
                # Extract a subgraph containing target word with full path (all the nodes) to it:
                targetSubGraph = getTargetWordSubGraphFullPath(inputGraph[0], targetNode)
                
                if summaryGraph == None:
                    summaryGraph = targetSubGraph
                else:
                    summaryGraph |= targetSubGraph
                    
            # Fix the formatting:
            summaryGraph = penman.encode(summaryGraph)
        
        # Use obtained AMR-graph to generate a summary:
        if summaryGraph != None:
            # Generate text from given AMR-graph:
            gtos = amrlib.load_gtos_model()
            #rulesEn, _ = gtos.generate([penman.encode(summaryGraph)])
            rulesEn, _ = gtos.generate([summaryGraph])
            
            # Fix potential translation errors (by removing "1." etc):
            rulesEn = [re.sub('\d. ', '', rulesEn[0])]
            
            # Translate it back to French:
            rulesFr = model.translate(rulesEn[0], source_lang='en', target_lang='fr')
            
            # Some post-processing:
            if "* * * * * *" in rulesFr:
                rulesFr = re.sub(' \*', '', rulesFr)
            
            # If there are several sentences, exclude the shortest one (except of some exceptions for which keep only the longest one):
            rulesFr_sent = sent_tokenize(rulesFr)
            if len(rulesFr_sent) > 1:
                if len(rulesFr_sent[1]) >= len(rulesFr_sent[0]):
                    rulesFr = rulesFr_sent[1]
                else:
                    rulesFr = rulesFr_sent[0]
                    
            # Additional post-processing:
            rulesFr = re.sub('\[\d', '', rulesFr)
            rulesFr = re.sub('2/3/4/5', 'N-2 N-3 N-4 N-5', rulesFr)
            if rulesFr[0] == "(" or rulesFr[0] == "*": 
                rulesFr = rulesFr[1:]
            
            # Save result to the file:
            f_out.write('> '+rulesFr+'\n\n')
        else:
            print('Error! Cannot find keywords in the graph')
            f_out.write('> Error! Cannot find keywords in the graph \n\n')

        f_out.write('\n')
        
f_out.close()
#%%
