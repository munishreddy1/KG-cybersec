
import os
import sys
import json
import pdb

path            = os.getcwd()
index           = path.rindex('/')
thoth_lab_path  = path[0:index]
chatbot_path    = thoth_lab_path + '/plugins/ChatBot/'

sys.path.insert(1, chatbot_path)

def get_response_from_intent(intent):

    json_obj = open(chatbot_path + 'response.json',)
    data     = json.load(json_obj)

    if (data[intent[0]]['textresponse'] != ''):
        data[intent[0]]['textFlag'] = 1
    else:
        data[intent[0]]['textFlag'] = 0

    #pdb.set_trace()
    #print('buttons' in data[intent[0]].keys())
    if 'buttons' in data[intent[0]].keys():
      if len(data[intent[0]]['buttons']) >= 1:
          data[intent[0]]['buttonFlag'] = 1
      else:
          data[intent[0]]['buttonFlag'] = 0
    return data[intent[0]]

if __name__ == '__main__':
    get_response_from_intent('metasploit')
