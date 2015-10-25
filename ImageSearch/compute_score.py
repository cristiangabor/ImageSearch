import pickle
import os
import sift1
import new_imagesearch
import vocabulary
from pysqlite2 import dbapi2 as sqlite
def get_imlist(path):
	
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.pgm')]

imlist=get_imlist('img')

newimlist=[]

for i in imlist:
	newimlist.append(i[4:])

with open('vocabulary.pkl','rb') as f:
	voc=pickle.load(f)


src=new_imagesearch.Searcher('test.db',voc)

print new_imagesearch.compute_ukbench_score(src,newimlist)


# show actual search

nbr_results=5

res=[w[1] for w in src.query(newimlist[0])[:nbr_results]]
print res
new_imagesearch.plot_results(src,res)

