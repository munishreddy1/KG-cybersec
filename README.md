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

## Reproducibility 
I tried to create the docker file in the repository as there is the requirements already existed in the main repository so that we can easyily rerun the system.
I am unable to run the docker container as my laptop is shutting down after running for long time and probably the resource issue. I hope the docker file will works better in better resource system. I will update the link of the repository with the docker file.
For time being I am not working not this as it is taking long hours to reproduce in my laptop.
We can try to work it on better resources laptop.
