from easydict import EasyDict as edict



f= {'a':1,'b':2}
f=edict(f)
#print(f['a'])

g = {'c':3}
g=edict(g)
f=g
print(f)