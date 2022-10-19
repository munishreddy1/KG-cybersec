# KG-Cybersec
In this project, we developed an ontology framework and knowledge graphs for teaching cybersecurity courses.
The data is available as unstructured text in lab manuals and course material. There is no standrad datasets available for cybersecurity.

## NER 
We used NER to extract the raw entities as subjects and objects in a sentence and relations as the root of the sentence. We store these as triples and generated prelim knowledge graphs from the extracted information. 

## Ontology Development
We then used domain knowledge to design an ontology framework for cybersecurity education and refined the extracted entities and relations. The key entity catgories and their types were identified. The key relations were also identified and an attribute called, 'action' was added to relations.
We then developed the knowledge graph from final triples.

## Entity Matcher
The custom entity matcher program can be used to run on other documents and identify the entities given in its KB file, kb_cyber.yaml. 
The lexical analyser lex_cyber.yaml helped in entity linking and scope resolution. The module Entity Matchher contains the code in pythonscrript Entity_matcher_cyberSec.py and the entity KB data is available in yaml files


## ChatBot
We built an intent classification chatbot using SVM based on key entities identified. The Module ChatBot contains the model, json file for responses and API implementation.
