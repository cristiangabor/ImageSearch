# CREATING A VOCABULARY

"""To create a vocabulary of visual words we first need to extract descriptors.
Here we use SIFT  descriptor. """
import os
import sift1
import vocabulary
import pickle
import new_imagesearch

def get_imlist(path):
	
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.pgm')]

imlist=get_imlist('img')

newimlist=[]

for i in imlist:
	newimlist.append(i[4:])


nbr_images=len(imlist)

featlist = [ imlist[i][:-3]+'sift' for i in range(nbr_images) ]

voc=vocabulary.Vocabulary('ukbenchtest')
voc.train(featlist,1000,10)


# saving vocabulary

with open('vocabulary.pkl','wb') as f:
	pickle.dump(voc,f)

print 'vocabulary is:', voc.name, voc.nbr_words
