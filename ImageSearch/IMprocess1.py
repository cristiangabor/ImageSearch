# CREATING A VOCABULARY

"""To create a vocabulary of visual words we first need to extract descriptors.
Here we use SIFT  descriptor. """
import os
import sift1
import vocabulary
import pickle
import imagesearch1

def get_imlist(path):
	'''Return a list of filenames for all jpg images
		in a directory'''

	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

imlist=get_imlist('pic')





nbr_images=len(imlist)

featlist = [ imlist[i][:-3]+'sift' for i in range(nbr_images) ]

voc=vocabulary.Vocabulary('ukbenchtest')
voc.train(featlist,1000,10)


# saving vocabulary

with open('vocabulary.pkl','wb') as f:
	pickle.dump(voc,f)

print 'vocabulary is:', voc.name, voc.nbr_words
#for i in range(nbr_images):
#	sift1.process_image(imlist[i],featlist[i])


src=imagesearch1.Searcher('test.db',voc)

#print featlist[0]
	
#locs,descr=sift1.read_features_from_file(featlist[0])

#iw=voc.project(descr)

#print 'ask using a histogram...'
#print src.candidates_from_histogram(iw)[:10]

print imlist[0]


print 'try a guery...'

print src.query(imlist[0])[:10]