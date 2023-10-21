#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[2]:


nlp= spacy.load('en_core_web_sm')


# In[3]:


from spacy.matcher import Matcher
from spacy.tokens import Span


# In[4]:


pd.set_option('display.max.colwidth',200)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


# import the lab 3 dataset
snortdata = pd.read_csv('/Users/garima/learningmodel/garima/dataset/lab3_dataset.csv')


# In[7]:


# test dataset
clouddata= pd.read_csv('/Users/garima/learningmodel/garima/dataset/cloudComputing_dataset.csv')


# In[8]:


clouddata.shape


# In[9]:


clouddata.head(10)


# In[12]:


#doc= nlp("Snort is a Intrusion Detection Systems")
#doc=nlp("Snort is a signature-based intrusion detection system used to detect network attacks")
#doc=nlp("lab3 has prereq deploy network test using hping3")
#doc= nlp("lab3 has prereq to use syslog for remote logging")
doc=nlp("Snort is used for signature based intrusion detection ")


# In[13]:


for tok in doc:
    print(tok.text, "--", tok.dep_)


# In[10]:


# entity pair extraction
def get_entities(sent):
    ent1= ""
    ent2= ""
    
    prv_tok_dep = "" # dependency tag of previous token in the sentence
    prv_tok_text = "" # previous token in the sentence
    
    
    prefix = ""
    modifier= ""
    
    
    
    for tok in nlp(sent):
        
        # if token is punctuation mark, move to next token
        if tok.dep_ != "punct":
            # check: token is compound or not
            if tok.dep_ == "compound" or tok.dep_ == "acl" or tok.dep_ == "prep":
                prefix = tok.text
                # if previous word is also a compound then add the cuurrentt word to it
                if prv_tok_dep == "compound":
                    prefix  = prv_tok_text + " " + tok.text
                    #print(prefix)
                # check: token is modifier or not
            if tok.dep_.endswith("mod")== True:
                modifier = tok.text
                    # if previous word was also a compound then add the current word to it
                if prv_tok_dep == "compound":
                    modifier  = prv_tok_text + " " + tok.text
                    #print(modifier)
                
                # check: if it is a subject - entity 1
            if tok.dep_.find("subj")== True:
                ent1= modifier + " " + prefix + " " + tok.text
                #print("the first entity is:",ent1)
                prefix = ""
                modifier= ""
                prv_tok_dep = "" 
                prv_tok_text= "" 
                    
                    
                    
                    
                # check if it is an object - entity 2
            if tok.dep_.find("obj")== True:
                ent2 = ent2 + " " + modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier= ""
                #print(ent2)
                    
                # update the variables  
            prv_tok_dep = tok.dep_ 
            prv_tok_text= tok.text
                
    return [ent1.strip(), ent2.strip()]
                


# In[7]:


entity_pairs=[]

for i in tqdm(snortdata["sentence"]):
    entity_pairs.append(get_entities(i))


# In[11]:


#entity_pairs for test data

entity_pairs_test=[]

for i in tqdm(clouddata["sentence"]):
    entity_pairs_test.append(get_entities(i))



# In[13]:


entitydf= pd.DataFrame(entity_pairs)


# In[12]:


# test entities
entitydf_test= pd.DataFrame(entity_pairs_test)


# In[14]:


entitydf.head()


# In[13]:


# test
entitydf_test.head()


# In[16]:


entitydf.to_csv('entity.csv')


# In[14]:


entitydf_test.to_csv('entity_test.csv')


# In[150]:


#entity_pairs=[]

get_entities("images are stored in S3 bucket")


# In[65]:


get_entities("deep learning model will be given to students")


# In[720]:


for tok in nlp("hping is sent from client to server"):
    print(tok.text, tok.dep_)


# In[15]:


# relation extraction
# predicate is the verb in a sentence
# write a rule to get the corresponding auxiliary and the root

def get_relation(sent):
    
    doc=nlp(sent)
    
    #matcher class object
    
    matcher = Matcher(nlp.vocab)
    
    #define the pattern
    
    pattern = [ {'POS':'AUX','OP':"*"},
               {'DEP': 'ROOT'},
              {'DEP': 'prep', 'OP':"?"},
              {'DEP':'agent', 'OP':"?"},
              {'POS':'ADJ','OP':"?"}]
    
    matcher.add("matching_1", [pattern], on_match=None)
    
    matches= matcher(doc)
    #print(matches)
    #print(len(matches))
    k=len(matches)-2
    #print(k)
    span= doc[matches[k][1]:matches[k][2]]
    
    return(span.text)
    


# In[151]:


get_relation("images are stored in S3 bucket")


# In[46]:


get_relation("Elastic application will offer cloud service")


# In[1158]:


get_relation("Snort can output tcpdump pcap")


# In[18]:


relations= [get_relation(i) for i in tqdm(snortdata["sentence"])]


# In[16]:


# test data

relations_test= [get_relation(i) for i in tqdm(clouddata["sentence"])]


# In[57]:


pd.Series(relations).value_counts()


# In[19]:


relationsdf = pd.DataFrame(relations)


# In[17]:


# relations for test data
relationsdf_test = pd.DataFrame(relations_test)


# In[21]:


relationsdf.to_csv('relations.csv')


# In[18]:


relationsdf_test.to_csv('relations_test.csv')


# In[22]:


# build the Knowledge Graph
# create a dataframe of entities and relations

# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})


# In[23]:


kg_df.to_csv('triples.csv')


# In[28]:


# create a directed graph from a dataframe

G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
#edge_labels= dict([((u,v), relations[d['weight']]) for u,v, d in G.edges(data=True)])


# In[ ]:


# create a directed graph from a dataframe  
   


# In[ ]:





# In[34]:


G = nx.DiGraph(directed=True)  
for i in range(len(relations)):
    G.add_weighted_edges_from([(source[i], target[i], i)])
        #print(G.nodes)
        #print(G.edges)
print("\n Knowledge Graph generated")
size =20
if (len(relations)/2)>20:
    size= len(edge)/2
plt.figure(figsize=(size, size))
edge_labels= dict([((u,v), relations[d['weight']]) for u,v, d in G.edges(data=True)])
pos=nx.spring_layout(G,k=0.8)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=5000, edge_color='r', edge_cmap=plt.cm.Blues, pos=pos, font_size=20)
nx.draw_networkx_edge_labels(G, pos, edge_labels= edge_labels, font_size=15)
plt.savefig('images/snort_req.png')


# In[35]:


#plot the graph

plt.figure(figsize=(12,12))
pos=nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
plt.show()


# In[39]:


# generate KG for snort lab requirements
# import the snort lab requuirement dataset
#snortreq = pd.read_csv('/Users/garima/learningmodel/garima/dataset/snort_req.csv')
#snortreq = snortdata[:11]

# generate KG for aws basic
# awsbasic= clouddata[:9]

# project 2 - intro
project2 = clouddata[34:]


# In[40]:


project2.head(10)


# In[41]:


entity_pairs=[]

for i in tqdm(project2["sentence"]):
    entity_pairs.append(get_entities(i))


# In[42]:


relations= [get_relation(i) for i in tqdm(project2["sentence"])]


# In[43]:


# build the Knowledge Graph
# create a dataframe of entities and relations

# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

#kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})


# In[45]:


G = nx.DiGraph(directed=True)
for i in range(len(relations)):
    G.add_weighted_edges_from([(source[i], target[i], i)])
#print(G.nodes)
#print(G.edges)
print("\n Knowledge Graph generated")
size =20
if (len(relations)/2)>20:
    size= len(edge)/2
plt.figure(figsize=(size, size))
edge_labels= dict([((u,v), relations[d['weight']]) for u,v, d in G.edges(data=True)])
pos=nx.spring_layout(G,k=0.8)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=5000, edge_color='r', edge_cmap=plt.cm.Blues, pos=pos, font_size=15)
nx.draw_networkx_edge_labels(G, pos, edge_labels= edge_labels, font_size=10)
plt.savefig('images/cloud_aws_project2.png')



# In[108]:


# generate KG for snort basics
# import the snort lab dataset
# snortintro=snortdata[9:40]
# snortintro.head(10)

# # generate KG for project 1
# clouddata= pd.read_csv('/Users/garima/learningmodel/garima/dataset/cloudComputing_dataset.csv')
# awsreq= clouddata[9:19]
# awsreq.head(20)

# generate KG for project 2 requirement
clouddata= pd.read_csv('/Users/garima/learningmodel/garima/dataset/cloudComputing_dataset.csv')
p2req= clouddata[43:]
p2req.head()


# In[109]:


entity_pairs=[]

for i in tqdm(p2req["sentence"]):
    entity_pairs.append(get_entities(i))


# In[110]:


relations= [get_relation(i) for i in tqdm(p2req["sentence"])]


# In[111]:


# build the Knowledge Graph
# create a dataframe of entities and relations

# extractt subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})


# In[115]:


# KG for snort introduction
G = nx.DiGraph(directed=True)
for i in range(len(relations)):
    G.add_weighted_edges_from([(source[i], target[i], i)])
#print(G.nodes)
#print(G.edges)
print("\n Knowledge Graph generated")
size =23
if (len(relations)/2)>23:
    size= len(edge)/2
plt.figure(figsize=(size, size))
edge_labels= dict([((u,v), relations[d['weight']]) for u,v, d in G.edges(data=True)])
pos=nx.spring_layout(G,k=0.8)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=5000, edge_color='r', edge_cmap=plt.cm.Blues, pos=pos, font_size=12)
nx.draw_networkx_edge_labels(G, pos, edge_labels= edge_labels, font_size=12)
plt.savefig('images/project2_req.png')




# In[100]:


# generate KG for snort basics
# import the snort lab dataset
# snorttask1=snortdata[41:47]
# snorttask1.head(10)
clouddata= pd.read_csv('/Users/garima/learningmodel/garima/dataset/cloudComputing_dataset.csv')
# awstask= clouddata[19:36]
# awstask.head(20)

p2task = clouddata[57:]
p2task.head(10)


# In[101]:


entity_pairs=[]

for i in tqdm(p2task["sentence"]):
    entity_pairs.append(get_entities(i))
    
relations= [get_relation(i) for i in tqdm(p2task["sentence"])]


# In[102]:


# build the Knowledge Graph
# create a dataframe of entities and relations

# extractt subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]


# In[137]:


# KG for snort introduction
G = nx.DiGraph(directed=True)
for i in range(len(relations)):
    G.add_weighted_edges_from([(source[i], target[i], i)])
#print(G.nodes)
#print(G.edges)
print("\n Knowledge Graph generated")
size =20
if (len(relations)/2)>20:
    size= len(edge)/2
plt.figure(figsize=(size, size))
edge_labels= dict([((u,v), relations[d['weight']]) for u,v, d in G.edges(data=True)])
pos=nx.spring_layout(G,k=0.8)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=4000, edge_color='r', edge_cmap=plt.cm.Blues, pos=pos, font_size=12)
nx.draw_networkx_edge_labels(G, pos, edge_labels= edge_labels, font_size=12)
plt.savefig('images/p2_task.png')


# ##### import the lab 3 dataset
# snortdata = pd.read_csv('/Users/garima/learningmodel/garima/dataset/lab3_dataset.csv')

# In[1388]:


snorttask2=snortdata[47:73]
snorttask2.head()


# In[1389]:


entity_pairs=[]

for i in tqdm(snorttask2["sentence"]):
    entity_pairs.append(get_entities(i))


# In[1390]:


relations= [get_relation(i) for i in tqdm(snorttask2["sentence"])]


# In[1391]:


# build the Knowledge Graph
# create a dataframe of entities and relations

# extractt subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]
kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})


# In[1401]:


# KG for snort introduction
G = nx.DiGraph(directed=True)
for i in range(len(relations)):
    G.add_weighted_edges_from([(source[i], target[i], i)])
#print(G.nodes)
#print(G.edges)
print("\n Knowledge Graph generated")
size =20
if (len(relations)/2)>20:
    size= len(edge)/2
plt.figure(figsize=(size, size))
edge_labels= dict([((u,v), relations[d['weight']]) for u,v, d in G.edges(data=True)])
pos=nx.spring_layout(G,k=0.8)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=5000, edge_color='r', edge_cmap=plt.cm.Blues, pos=pos, font_size=15)
nx.draw_networkx_edge_labels(G, pos, edge_labels= edge_labels, font_size=15)
plt.savefig('images/snort_task2.png')



# In[1288]:


# import the lab 3 dataset
snortdata = pd.read_csv('/Users/garima/learningmodel/garima/dataset/lab3_dataset.csv')


# In[1290]:


snorttask3=snortdata[97:116]
snorttask3.head(30)


# In[1291]:


entity_pairs=[]

for i in tqdm(snorttask3["sentence"]):
    entity_pairs.append(get_entities(i))


# In[1292]:


relations= [get_relation(i) for i in tqdm(snorttask3["sentence"])]


# In[1293]:


# build the Knowledge Graph
# create a dataframe of entities and relations

# extractt subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})


# In[1295]:


# KG for snort introduction
G = nx.DiGraph(directed=True)
for i in range(len(relations)):
    G.add_weighted_edges_from([(source[i], target[i], i)])
#print(G.nodes)
#print(G.edges)
print("\n Knowledge Graph generated")
size =20
if (len(relations)/2)>20:
    size= len(edge)/2
plt.figure(figsize=(size, size))
edge_labels= dict([((u,v), relations[d['weight']]) for u,v, d in G.edges(data=True)])
pos=nx.spring_layout(G,k=0.8)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=5000, edge_color='r', edge_cmap=plt.cm.Blues, pos=pos, font_size=15)
nx.draw_networkx_edge_labels(G, pos, edge_labels= edge_labels, font_size=12)
plt.savefig('images/snort_task3.png')



# In[1358]:


# import the lab 3 dataset
snortdata = pd.read_csv('/Users/garima/learningmodel/garima/dataset/lab3_dataset.csv')


# In[1361]:


snorttask4=snortdata[116:128]
snorttask4.head(20)


# In[1362]:


entity_pairs=[]

for i in tqdm(snorttask4["sentence"]):
    entity_pairs.append(get_entities(i))


# In[1363]:


relations= [get_relation(i) for i in tqdm(snorttask4["sentence"])]


# In[1364]:


# build the Knowledge Graph
# create a dataframe of entities and relations

# extractt subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]


# In[1365]:


# KG for snort introduction
G = nx.DiGraph(directed=True)
for i in range(len(relations)):
    G.add_weighted_edges_from([(source[i], target[i], i)])
#print(G.nodes)
#print(G.edges)
print("\n Knowledge Graph generated")
size =20
if (len(relations)/2)>20:
    size= len(edge)/2
plt.figure(figsize=(size, size))
edge_labels= dict([((u,v), relations[d['weight']]) for u,v, d in G.edges(data=True)])
pos=nx.spring_layout(G,k=0.8)
nx.draw(G, with_labels=True, node_color='lightblue', node_size=5000, edge_color='r', edge_cmap=plt.cm.Blues, pos=pos, font_size=15)
nx.draw_networkx_edge_labels(G, pos, edge_labels= edge_labels, font_size=12)
plt.savefig('images/snort_task4.png')


# In[ ]:




