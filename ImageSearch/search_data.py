import os
import sift1
import new_imagesearch
import vocabulary

def get_imlist(path):
	
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.pgm')]

imlist=get_imlist('img')

newimlist=[]

nbr_images=len(imlist)

featlist= [ imlist[i][:-3]+'sift' for i in range(nbr_images) ]

voc=vocabulary.Vocabulary('ukbenchtest')
voc.train(featlist,1000,10)

src=new_imagesearch.Searcher('test.db',voc)

locs,descr = sift1.read_features_from_file(featlist[0])
iw= voc.project(descr)

print 'ask using histogram...'
print src.candidates_from_histogram(iw)[:10]


