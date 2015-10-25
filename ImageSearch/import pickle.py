import pickle
import os
import sift1
import new_imagesearch
import homography


# load  image list and vocabulary
def get_imlist(path):
	
	return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.pgm')]

imlist=get_imlist('img')

newimlist=[]

for i in imlist:
	newimlist.append(i[4:])

nbr_images=len(imlist)
featlist = [ imlist[i][:-3]+'sift' for i in range(nbr_images) ]




with open('vocabulary.pkl','rb') as f:
	voc= pickle.load(f)


src=new_imagesearch.Searcher('test.db',voc)

# index of query image and number of results to return

q_ind=0
nbr_results=8


# regular query

res_reg = [w[1] for w in src.query(newimlist[q_ind])[:nbr_results]]
print 'top matches (regular):',res_reg

print newimlist[1]
# load image features from query image

q_locs,q_descr=sift1.read_features_from_file(featlist[q_ind])

fp=homography.make_homog(q_locs[:,:2].T)

# RANSAC model for homography fitting

model = homography.RansacModel()

rank={}

# load image features for result

for ndx in res_reg[1:]:
	print ndx
	locs,descr = sift1.read_features_from_file(featlist[ndx])


	# get matches

	matches=sift1.match(q_descr,descr)
	ind =matches.nonzero()[0]
	ind2 =matches[ind]
	tp = homography.make_homog(locs[:,:2].T)

	# compute homography,count inliers, if not enough matches return empty list

	try:
		H,inliers = homography.H_from_ransac(fp[:,ind],tp[:,ind2],model,match_theshold=4)

	except:
		inliers =[]

	# store inlier count

	rank[ndx] = len(inliers)


# sort dictionary to get the most inliers first

sorted_rank=sorted(rank.items(),key=lambda t: t[1],reverse=True)
res_geom=[res_reg[0]] + [s[0] for s in sorted_rank]

print 'top matches (homography):',res_geom

# plot the top results

new_imagesearch.plot_results(src,res_reg[:4])
new_imagesearch.plot_results(src,res_geom[:4])