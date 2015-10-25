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


nbr_images=len(imlist)

featlist= [ imlist[i][:-3]+'sift' for i in range(nbr_images) ]

voc=vocabulary.Vocabulary('ukbenchtest')
voc.train(featlist,1000,10)

src=new_imagesearch.Searcher('test.db',voc)

print 'try a query...'
print src.query(newimlist[0])[:10]