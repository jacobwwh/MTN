import sys
import os
import os.path
import pycparser
from pycparser import c_parser
from ast2struct import ast2list

allast=[]

def get_token(node, lower=True):
    name = node.__class__.__name__
    token = name
    is_name = False
    if len(node.children()) == 0:
        attr_names = node.attr_names
        if attr_names:
            if 'names' in attr_names:
                token = node.names[0]
            elif 'name' in attr_names:
                token = node.name
                is_name = True
            else:
                token = node.value
        else:
            token = name
    else:
        if name == 'TypeDecl':
            token = node.declname
        if node.attr_names:
            attr_names = node.attr_names
            if 'op' in attr_names:
                if node.op[0] == 'p':
                    token = node.op[1:]
                else:
                    token = node.op
    if token is None:
        token = name
    if lower and is_name:
        token = token.lower()
    return token
  
def token2dict():
    allalldecl=[]
    alltokenlist=[]
    def visitast(node, mynodename=None):
        alltokenlist.append(get_token(node)) #add identifier name
        alltokenlist.append(node.__class__.__name__) #add node type
        for (child_name, child) in node.children():
            visitast(child, mynodename=child_name)

    parser = c_parser.CParser()
    for i in range(1, 294):
        dirname = 'poj200/' + str(i) + '/'
        #print(dirname)
        for rt, dirs, files in os.walk(dirname):
            for file in files:
                filename = os.path.join(rt, file)
                file = open(filename, mode='rt')
                code = file.read()
                code = code.split('\r')
                newcode = ''
                for line in code:
                    newcode = newcode + line
                file.close()
                code=newcode
                ast = parser.parse(code)
                visitast(ast)
    alltokenlist = list(set(alltokenlist))

    #print(alltokenlist)
    alltokenlist.append('UNK')
    #print(len(alltokenlist))
    vocablen=len(alltokenlist)
    vocabdict = dict(zip(alltokenlist, range(vocablen)))
    #print(vocabdict)
    return vocabdict, vocablen
