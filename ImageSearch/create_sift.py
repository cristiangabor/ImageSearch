import os 
import sift1

def get_path(path):

	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]



imlist=get_path('img')

nbr_images=len(imlist)

featlist = [ imlist[i][:-3] +'sift' for i in range(nbr_images)]

for i in range(nbr_images):
	sift1.process_image(imlist[i],featlist[i])