#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 23:16:25 2022

@author: garima
"""
import spacy
from spacy import displacy
import yaml
import os

nlp= spacy.load('en_core_web_sm')

# create a simple is-a taxonomy
from typing import List, Tuple

path=os.getcwd()

with open (r'/Users/garima/learningmodel/garima/EntityMatcher/kb_cyber.yaml') as file:
    kb_dat= yaml.load(file, Loader=yaml.FullLoader)
  #  print(kb_dat['THINGS'])
  #  print(kb_dat['RELS'])


class Thing(object):
    
    def __init__(self, uri:str, description:str=None):
        self.uri=uri
        self.description=description
        
    def __repr__(self):
        return self.uri
    
class KB(object):
    
    def __init__(self, name, things: List[Thing], is_a_relations: List[Tuple[str,str]]):
        self.name=name
        self.things= {thing.uri: thing for thing in things}
        self.rels=is_a_relations
        
    def get_thing(self, uri:str):
        return self.thing.get(uri,None)
    
    def get_parent(self, uri:str) -> List[str]:
        for r in rels:
            if r[1]==uri:
                return r[0]
        return None 
    
    def get_children(self, uri:str)-> List[str]:
        return tuple([r[1] for r in rels if r[0]==uri])
    
    def __repr__(self) -> str:
        header = f"---- KB {self.name} ----"
        obj_str = [header]
        obj_str.append("My things:")
        for thing in self.things:
            obj_str.append(f"\t{thing}")
        obj_str.append("\nMy relations")
        for rel in self.rels:
            obj_str.append(f"\t{rel[1]} is-a {rel[0]}")
        obj_str.append("-"*len(header))
        return "\n".join(obj_str)
    
things = [Thing(f"{kb_dat['URI_PREFIX']}/{name}") for name in kb_dat['THINGS']]
rels = [(f"{kb_dat['URI_PREFIX']}/{parent}", f"{kb_dat['URI_PREFIX']}/{child}") for parent, children in kb_dat['RELS'].items() 
        for child in children]

kb = KB("basic_ner", things, rels)
#print(kb)
        
#create Lexical KB
class LexEntry(object):
    def __init__(self, uri, canonical_form, other_forms):
        self.uri = uri
        self.canonical_form = canonical_form
        self.other_forms = other_forms
        
    def get_forms(self):
        return [self.canonical_form] + self.other_forms[:]
        
    def __repr__(self):
        return f"{self.uri}\t<->\t{self.canonical_form} : {self.other_forms}"
    
class LexKB(object):
    def __init__(self, name, lex_entries: List[LexEntry]):
        self.name = name
        self.lex_entries = lex_entries
        self.meaning2forms = None
        self.form2meanings = None
        self.setup()
       
    def setup(self) -> None:
        self.meaning2forms = {le.uri: [le.canonical_form] + le.other_forms for le in self.lex_entries}
        self.form2meanings = {}
        for le in self.lex_entries:
            for form in le.get_forms():
                if form in self.form2meanings:
                    self.form2meanings[form].append(le.uri)
                else:
                    self.form2meanings[form] = [le.uri]
                           
    def get_meanings(self, form):
        return set(self.form2meanings.get(form, []))
    
    def get_forms(self, uri):
        return set(self.meaning2forms.get(uri, []))
    
    def __repr__(self):
        header = f"------ LEXICAL KB  {self.name} ------"
        my_str = [header] + [str(le) for le in self.lex_entries] + ["-"*len(header)]
        return "\n".join(my_str)  

# open the lex dat file
with open (r'/Users/garima/learningmodel/garima/EntityMatcher/lex_cyber.yaml') as file:
    lex_dat= yaml.load(file, Loader=yaml.FullLoader)

    print(lex_dat['Snort'])
    print(lex_dat['IDS'])


lex_entries = [LexEntry(f"{kb_dat['URI_PREFIX']}/{name}", forms[0], forms[1:]) for name, forms in lex_dat.items()]

lexkb = LexKB("BasicNERLex", lex_entries)

print(lexkb)

from pathlib import Path
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span
from spacy import Language

class LexKBMatcher(object):
    name = 'lexkbmatcher'
    extension = 'kb_ner'
    def __init__(self, lexkb: LexKB, kb):
        self.lexkb = lexkb
        self.kb = kb
        self.matcher = None
        
    def make_matcher(self, nlp, attr: str ='ORTH') -> None:
        Doc.set_extension(self.extension, default=[], force=True)
        print("Initializing PhraseMatcher...")
        matcher = PhraseMatcher(nlp.vocab, attr=attr)
        for le in self.lexkb.lex_entries:
            forms = [nlp(form.lower()) for form in le.get_forms()]
            matcher.add(le.uri, None, *forms)
        self.matcher = matcher
        print("Done!")
        
    def _simplify_uri(self, uri):
        return Path(uri).name
        
    def __call__(self, doc):
        matches = self.matcher(doc)
        res_spans = []
        for match_id, start, end in matches:
            uri = doc.vocab.strings[match_id]
            label = self._simplify_uri(kb.get_parent(uri))
            span = Span(doc, start, end, label=label, kb_id=uri)
            res_spans.append(span)
        doc._.set(self.extension, res_spans)
        return doc
        
    
lexkb_matcher = LexKBMatcher(lexkb, kb)
lexkb_matcher.make_matcher(nlp, 'ORTH')

text = """One planet, dozens of cities and 10 million Californians collectively 
benefited during the past decade from the creation of community choice aggregators, 
or CCAs, in local communities throughout the state.
More commonly known as community choice energy programs, locally controlled CCAs give residents and businesses 
the ability to select greener, renewable sources of electricity often at a lower cost than provided by 
California’s three investor-owned utilities: Pacific Gas and Electric Co., San Diego Gas & Electric Co., 
and Southern California Edison."""

text="Good morning Danny. This is John from california at North Carolina Electric Membership Corporation and Duke Energy, it is June 5, 2021 today"
#doc = nlp(text)
#displacy.render(doc, style='ent')

## Expect lowercased text ....
doc = nlp(text.lower())
doc = lexkb_matcher(doc)
doc1 = nlp(text)

for ent in doc1.ents:
    print(ent.text, ' ', ent.label_)

for ent in doc._.kb_ner:
    print(ent, ent.start, ent.end)

def render(doc):  
    ents = [{'name': span, 'label': span.label_} for span in doc._.kb_ner]
   
    filtered_ents  = []
    
    for ent in ents:
        filtered_ents.append(ent)
       
    dat = {'text': doc.text, 'ents': filtered_ents}
    #displacy.render(dat, style='ent', manual=True)
    return dat
    
    
utility_list=render(doc, doc1)

#print(utility_list)

def render_all(doc, doc1):
    ents = [{'name': span, 'label': span.label_} for span in doc._.kb_ner]
    ents += [{'name': ent,  'label': ent.label_} for ent in doc.ents]
    for ent in doc1.ents:
    
        if ent.label_=="PERSON":
            
            ents += [{'name': ent,  'label': ent.label_}]
            
    filtered_ents  = []
   
    for ent in ents:
        filtered_ents.append(ent)
    dat = {'text': doc.text, 'ents': filtered_ents}
    return dat
   
entity_all_list = render_all(doc, doc1) 

print(entity_all_list)
    