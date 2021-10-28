#mtn-b
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import sys
import pycparser
from tqdm import tqdm,trange
sys.path.append('../')
from createdataset import createdata
from trainw2v import token2dict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=100)
args = parser.parse_args()
num_epochs=50
embedding_dim=args.embedding_dim
batch_size=32
clip=0.5
dropout_rate=0.5
dropout=nn.Dropout(dropout_rate)

vocabdict,vocablen=token2dict()

CUDA=True
def Var(v):
    if CUDA: return Variable(v.cuda())
    else: return Variable(v)

empty=np.zeros((embedding_dim,),dtype='float32')
empty=Var(torch.LongTensor(empty))

class funcdeclblock(nn.Module):
    def __init__(self, dim):
        super(funcdeclblock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim,bias=True)

    def forward(self, x, y):
        x=x.view(-1)
        y=y.view(-1)
        out = torch.cat([x, y], 0)  # Concatentate along depth
        out = F.relu(self.proj(out))
        return out

class funccallblock(nn.Module):
    def __init__(self, dim):
        super(funccallblock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim,bias=True)

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        out = torch.cat([x, y], 0)  # Concatentate along depth
        out = F.relu(self.proj(out))
        return out
      
class whileblock(nn.Module):
    def __init__(self, dim):
        super(whileblock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim,bias=True)

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        out = torch.cat([x, y], 0)  # Concatentate along depth
        out = F.relu(self.proj(out))
        return out

class dowhileblock(nn.Module):
    def __init__(self, dim):
        super(dowhileblock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim,bias=True)

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        out = torch.cat([x, y], 0)  # Concatentate along depth
        out = F.relu(self.proj(out))
        return out
     
class forblock(nn.Module):
    def __init__(self, dim):
        super(forblock, self).__init__()
        self.proj1 = nn.Linear(3 * dim, dim,bias=True)
        self.proj2 = nn.Linear(2 * dim, dim, bias=True)

    def forward(self, init,cond,next,stmt):
        init = init.view(-1)
        cond = cond.view(-1)
        next = next.view(-1)
        stmt = stmt.view(-1)
        x = torch.cat([init,cond,next], 0)
        x = F.relu(self.proj1(x))
        y=torch.cat([x,stmt],0)
        y = F.relu(self.proj2(y))
        return y
      
class ifblock(nn.Module):
    def __init__(self, dim):
        super(ifblock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim,bias=True)
        self.proj2 = nn.Linear(2 * dim, dim, bias=True)

    def forward(self, x, y,z=None,numnodes=2):
        x = x.view(-1)
        y = y.view(-1)
        out1 = torch.cat([x, y], 0)
        out1 = F.relu(self.proj(out1))
        if numnodes==2: #no else statement
            return out1
        z=z.view(-1)
        out2 = torch.cat([x, z], 0)
        out2 = F.relu(self.proj(out2))
        out=torch.max(out1,out2)
        return out
      
class switchblock(nn.Module):
    def __init__(self, dim):
        super(switchblock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim,bias=True)

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        out = torch.cat([x, y], 0)  # Concatentate along depth
        out = F.relu(self.proj(out))
        return out

class caseblock(nn.Module):
    def __init__(self, dim):
        super(caseblock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim,bias=True)

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        out = torch.cat([x, y], 0)  # Concatentate along depth
        out = F.relu(self.proj(out))
        return out
      
class seqblock(nn.Module):
    def __init__(self, dim):
        super(seqblock, self).__init__()
        self.hidden_dim = dim
        self.lstm = nn.LSTM(dim, dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Var(torch.zeros(1, 1, self.hidden_dim)),
                Var(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, (self.hidden[0].detach(),self.hidden[1].detach()))
        out=lstm_out[-1]
        out=out.view(-1)
        return out
      

class funcdecllstm(nn.Module):
    def __init__(self, dim,mem_dim):
        super(funcdecllstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self,inputs, child_c, child_h,child_h_sum):
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c,h
      
class whilelstm(nn.Module):
    def __init__(self, dim,mem_dim):
        super(whilelstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self,inputs, child_c, child_h,child_h_sum):
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c,h
      
class dowhilelstm(nn.Module):
    def __init__(self, dim,mem_dim):
        super(dowhilelstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self,inputs, child_c, child_h,child_h_sum):
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c,h
      
class forlstm(nn.Module):
    def __init__(self, dim,mem_dim):
        super(forlstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self,inputs, child_c, child_h,child_h_sum):
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c,h
      
class iflstm(nn.Module):
    def __init__(self, dim,mem_dim):
        super(iflstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self,inputs, child_c, child_h,child_h_sum):
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c,h
      
class switchlstm(nn.Module):
    def __init__(self, dim,mem_dim):
        super(switchlstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self,inputs, child_c, child_h,child_h_sum):
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c,h
      
class caselstm(nn.Module):
    def __init__(self, dim,mem_dim):
        super(caselstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self,inputs, child_c, child_h,child_h_sum):
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c,h
      
class seqlstm(nn.Module):
    def __init__(self, dim,mem_dim):
        super(seqlstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self,inputs, child_c, child_h,child_h_sum):
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c,h
      
class otherlstm(nn.Module):
    def __init__(self, dim,mem_dim):
        super(otherlstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self,inputs, child_c, child_h,child_h_sum):
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c,h
      

def get_token(node, lower=True): #get identifier name
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
  
  
class treelstm(nn.Module): # the MTN classifier
    def __init__(self, dim,mem_dim):
        super(treelstm, self).__init__()
        self.dim=dim
        self.mem_dim = mem_dim

        self.fdeclb=funcdeclblock(dim=dim)
        self.fcallb=funccallblock(dim=dim)
        self.whileb=whileblock(dim=dim)
        self.dowhileb=dowhileblock(dim=dim)
        self.forb=forblock(dim=dim)
        self.ifb=ifblock(dim=dim)
        self.switchb=switchblock(dim=dim)
        self.caseb=caseblock(dim=dim)
        self.seqb=seqblock(dim=dim)
        self.fdecll = funcdecllstm(dim, mem_dim)
        self.whilel = whilelstm(dim, mem_dim)
        self.dowhilel = dowhilelstm(dim, mem_dim)
        self.forl = forlstm(dim, mem_dim)
        self.ifl = iflstm(dim, mem_dim)
        self.switchl = switchlstm(dim, mem_dim)
        self.casel = caselstm(dim, mem_dim)
        self.seql = seqlstm(dim, mem_dim)
        self.otherl = otherlstm(dim, mem_dim)
        self.fc=nn.Linear(dim,1)
        self.nclass=293 #number of program classes
        self.fc2=nn.Linear(dim,self.nclass)
        self.embeddings=nn.Embedding(vocablen,embedding_dim)


    def node_forward(self, inputs, child_c, child_h,nodetype,childisseq):
        if type(child_h)==list:
            if nodetype=='FuncDef':
                child_h_sum=self.fdeclb(child_h[0],child_h[1])
                child_h = torch.cat(child_h)
                child_h = F.torch.unsqueeze(child_h, 1)
            elif nodetype=='While':
                child_h_sum = self.whileb(child_h[0], child_h[1])
                child_h = torch.cat(child_h)
                child_h = F.torch.unsqueeze(child_h, 1)
            elif nodetype=='DoWhile':
                child_h_sum=self.dowhileb(child_h[0], child_h[1])
                child_h = torch.cat(child_h)
                child_h = F.torch.unsqueeze(child_h, 1)
            elif nodetype=='Switch':
                child_h_sum=self.switchb(child_h[0], child_h[1])
                child_h = torch.cat(child_h)
                child_h = F.torch.unsqueeze(child_h, 1)
            elif nodetype=='For':
                if len(child_h)==4: #'For' nodes do not always have 4 children
                    child_h_sum=self.forb(child_h[0],child_h[1],child_h[2],child_h[3])
                elif len(child_h)==3:
                    child_h_sum=self.forb(self.embeddings(Var(torch.LongTensor([vocabdict['UNK']]))),child_h[0],child_h[1],child_h[2])
                elif len(child_h) == 2:
                    child_h_sum = self.forb(self.embeddings(Var(torch.LongTensor([vocabdict['UNK']]))), child_h[0],
                                            self.embeddings(Var(torch.LongTensor([vocabdict['UNK']]))), child_h[1])
                elif len(child_h) == 1:
                    child_h_sum=child_h[0].view(-1)
                child_h = torch.cat(child_h)
                child_h = F.torch.unsqueeze(child_h, 1)
            elif nodetype=='If':
                if len(child_h)==2:
                    child_h_sum=self.ifb(child_h[0],child_h[1],z=None,numnodes=2)
                elif len(child_h)==3:
                    child_h_sum=self.ifb(child_h[0],child_h[1],child_h[2],numnodes=3)
                child_h = torch.cat(child_h)
                child_h = F.torch.unsqueeze(child_h, 1)
            elif nodetype=='Case':
                stmtsid = child_h[1:]
                if stmtsid == []:
                    # print('case empty!')
                    child_h_sum = child_h[0].view(-1)
                else:
                    stmts = []
                    for child in stmtsid:
                        stmts.append(child)
                    stmts = torch.cat(stmts).view(len(stmts), 1, -1)
                    stmtfinal = self.seqb(stmts).view(-1)
                    child_h_sum = self.caseb(child_h[0], stmtfinal)
                child_h = torch.cat(child_h)
                child_h = F.torch.unsqueeze(child_h, 1)
            else:
                if childisseq==True:
                    child_h=torch.cat(child_h)
                    child_h = F.torch.unsqueeze(child_h, 1)
                    child_h_sum=self.seqb(child_h)
                else:
                    child_h = torch.cat(child_h)
                    child_h = F.torch.unsqueeze(child_h, 1)
                    child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)
        else:
            child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)
        child_c=F.torch.unsqueeze(child_c, 1)

        if nodetype == 'FuncDef':
            c, h = self.fdecll(inputs, child_c, child_h, child_h_sum)
        elif nodetype == 'While':
            c, h = self.whilel(inputs, child_c, child_h, child_h_sum)
        elif nodetype == 'DoWhile':
            c, h = self.dowhilel(inputs, child_c, child_h, child_h_sum)
        elif nodetype == 'Switch':
            c, h = self.switchl(inputs, child_c, child_h, child_h_sum)
        elif nodetype == 'For':
            c, h = self.forl(inputs, child_c, child_h, child_h_sum)
        elif nodetype == 'If':
            c, h = self.ifl(inputs, child_c, child_h, child_h_sum)
        elif nodetype == 'Case':
            c, h = self.casel(inputs, child_c, child_h, child_h_sum)
        else:
            if childisseq == True:
                c, h = self.seql(inputs, child_c, child_h, child_h_sum)
            else:
                c, h = self.otherl(inputs, child_c, child_h, child_h_sum)
        return c, h

    def traverse(self, tree):
        currentnode = Var(torch.FloatTensor(self.dim))
        currentinput = Var(torch.FloatTensor(self.dim))
        currentnodetype = tree.__class__.__name__
        word = currentnodetype
        #word = get_token(tree) #if you want to use identifier names, uncomment this line
        currentinput = self.embeddings(Var(torch.LongTensor([vocabdict[word]])))
        seqistrue = False
        for (child_name, child) in tree.children():
            if child_name.find('[') != -1:
                seqistrue = True
                break
        if tree.children() == ():
            child_c = Var(torch.zeros(1,  self.mem_dim))
            child_h = Var(torch.zeros(1, self.mem_dim))
        else:
            childcs = []
            childhs=[]
            for (child_name, child) in tree.children():
                c,h=self.traverse(child)
                childcs.append(c)
                childhs.append(h)
            child_c=torch.cat(childcs)
            child_h=childhs
        currentc,currenth = self.node_forward(currentinput, child_c, child_h,currentnodetype,seqistrue)
        return currentc,currenth

    def forward(self, x):
        out = self.traverse(x)[1]
        out=out.view(-1)
        out=self.fc2(out)
        return out
      
      
model=treelstm(embedding_dim,embedding_dim)
criterion = nn.CrossEntropyLoss()
if CUDA==True:
    model=model.cuda()
    criterion=criterion.cuda()
    
optimizer=optim.Adam(model.parameters(),lr=0.001)

trainset,validset,testset=createdata()
traindata,trainlabel=zip(*trainset)
validdata,validlabel=zip(*validset)
testdata,testlabel=zip(*testset)

def test(data,labels):
    time_start = time.time()
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(len(data)):
            inputdata = data[i]
            label = Var(torch.LongTensor(np.array([labels[i]])))
            output = model(inputdata)
            output = output.view(1, -1)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label.data).sum()
    time_end = time.time()
    print('val time for 1 epoch: ', time_end - time_start)
    print('acc: ' + str(float(correct) / float(len(data))))
    acc=float(correct)/float(len(data))
    return acc
  
bestacc=0
bestepoch=0
print('mtn-b')
for epoch in range(num_epochs):
    model.train()
    print('epoch: '+str(epoch+1))
    random.shuffle(trainset)
    traindata, trainlabel = zip(*trainset)
    optimizer.zero_grad()
    loss=0.0
    k=0
    totalloss = 0.0
    print(len(traindata))
    for i in range(len(traindata)):
        inputdata=traindata[i]
        label=Var(torch.LongTensor(np.array([trainlabel[i]])))
        output=model(inputdata)
        output=output.view(1,-1)
        _, predicted = torch.max(output.data, 1)
        err=criterion(output,label)
        loss+=err.item()
        totalloss += err.item()
        err.backward(retain_graph=True)
        k+=1
        if k%batch_size==0:
            optimizer.step()
            optimizer.zero_grad()
            sys.stdout.write('{0} batchloss: {1}\r'.format(k,loss))
            sys.stdout.flush()
            loss=0.0
    avgloss = totalloss / len(traindata)
    print('avgloss: ', avgloss)
    if epoch%10==9:
        print('train set: ')
        trainacc=test(traindata,trainlabel)
    print('valid set: ')
    devacc=test(validdata,validlabel)
    if devacc>bestacc:
        bestacc=devacc
        bestepoch=epoch+1
        torch.save(model,'mtnb'+str(embedding_dim)+'.pt')
    print('best epoch:',bestepoch)
    print('test set: ')
    testacc=test(testdata,testlabel)
