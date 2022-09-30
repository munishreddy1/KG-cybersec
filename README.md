# KG-Cybersec
In this project, we developed an ontology framework and knowledge graphs for teaching cybersecurity courses.
The data is available as unstructured text in lab manuals and course material. There is no standrad datasets available for cybersecurity.
We used NER to extract the entities as subjects and objects in a sentence and relations as the root of the sentence. We store these as triples and generated preim knowledge graphs from the extracted information. 
We then used domain knowledge to design an ontology framework for cybersecurity education and refined the extracted entities and relations. The key entity catgories and their types were identified. The key relations were also identified and an attribute called, 'action' was added to relations.
We then developed the knowledge graph from final triples.

The custom entity matcher program can be used to run on otherr documents and identify the entities given in its KB file, kb_cyber.yaml. 
The lexical analyser lex_cyber.yaml helped in entity linking and scope resolution.
We built an intent classification chatbot using SVM based on key entities identified.
