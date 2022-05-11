import os
import re
import random
import pycparser
from pycparser import c_parser

def createdata():
    trainset=[]
    validset=[]
    testset=[]
    parser = c_parser.CParser()
    for i in range(1, 294):
        dirname = 'poj200/' + str(i) + '/'
        j=0
        for rt, dirs, files in os.walk(dirname):
            for file in files:
                filename=dirname+file
                file=open(filename)
                code=file.read()
                code = code.split('\r')
                newcode = ''
                for line in code:
                    newcode = newcode + line
                code=newcode
                ast=parser.parse(code)
                #print(filename)
                if j%10==8:
                    validset.append([ast,i-1])
                elif j%10==9:
                    testset.append([ast,i-1])
                else:
                    trainset.append([ast,i-1])
                j+=1
                file.close()
    #random.shuffle(trainset)
    #random.shuffle(validset)
    #random.shuffle(testset)
    print(len(trainset),len(validset),len(testset))
    return trainset,validset,testset
