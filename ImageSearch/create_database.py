import pickle
import os
import sift1
import new_imagesearch
import vocabulary

def get_imlist(path):
	
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.pgm')]

imlist=get_imlist('img')
nbr_images = len(imlist)
featlist = [ imlist[i][:-3]+'sift' for i in range(nbr_images) ]

newimlist=[]
for i in imlist:
	newimlist.append(i[4:])



# load vocabulary

with open('vocabulary.pkl','rb') as f:
	voc=pickle.load(f)


# create indexer

indx = new_imagesearch.Indexer('test.db',voc)
"""Uncomment when you need to recreate the database"""

#indx.create_tables()  

""" End"""

# Add images  to database
for i in range(nbr_images):
	locs,descr=sift1.read_features_from_file(featlist[i])
	indx.add_to_index(newimlist[i],descr)

indx.db_commit()