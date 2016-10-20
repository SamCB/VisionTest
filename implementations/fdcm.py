#!/usr/bin/env python

import numpy as np
import cv2
from scipy.ndimage.morphology import distance_transform_edt
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from imutils import resize
import time

def sliding_window(x, wshape, wsteps):
    x = np.asarray(x)
    size = x.itemsize
    h, w = x.shape
    wh, ww = wshape
    sh, sw = wsteps

    out_shape = ((h-wh+1)/sh, (w-ww+1)/sw, wh, ww)
    out_strides = size*np.array([w*sh, sw, w, 1])
    
    return np.lib.stride_tricks.as_strided(x, out_shape, out_strides)

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def vectorized_template_average(dist_map, template, steps=(1,1)):
	sw = sliding_window(distance_transform_edt(dist_map), template.shape, steps)
	#get the positions of edge pixels on the template and reshape into an array of points
	idx = np.where(template>0)
	idx = np.hstack(( idx[0][:,None], idx[1][:,None] ))

	#get a combination of all values for the x, y in (y, x, wy, wx)
	swid = cartesian((np.arange(sw.shape[0]), np.arange(sw.shape[1])))
	#once we have x, y points add to it the wy wx points to form a single tuple
	idsw = np.hstack((np.repeat(swid, idx.shape[0], axis=0), np.tile(idx, (swid.shape[0],1))))
	#use the idx created above and index the sliding window
	tval = sw[(idsw[:,0],idsw[:,1],idsw[:,2],idsw[:,3])].reshape(sw.shape[0], sw.shape[1], -1)
	#average along the third axis
	return np.average(tval, axis=2)

def orientation_map(image, edge_map):
	Sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, 3)
	Sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, 3)

	ori = cv2.phase(Sx, Sy, True)
	ori[edge_map.astype(bool)==False] = -1.

	return ori

def normalize(x, bins=None):
	x = x.astype(float)
	x += max(abs(np.min(x)), 1.)
	x /= np.max(x)

	if bins:
		x *= bins
		return x.astype(int)
	else:
		return x

def auto_canny(image, sigma=0.33):
		v = np.median(image)
		lower = int(max(0,(1.0-sigma) * v))
		upper = int(min(255, (1.0+sigma) * v))
		edged = cv2.Canny(image, lower, upper)
		return edged

def heatmap(image):
	x = normalize(image, bins=255).astype(np.uint8)
	x = cv2.applyColorMap(x, 2)
	return x
    
if __name__ == '__main__':
	#video = cv2.VideoCapture('raw_images\\%04d.bmp')
	video = cv2.VideoCapture(0)

	video.set(3, 160)
	video.set(4, 120)

	step_size = 2
	#t = np.zeros((50,50), np.uint8)
	#t = cv2.circle(t, (25,25), 24, (255,255,255), 1)

	twidth = 40
	timage = np.zeros((twidth,twidth), np.uint8)
	timage = cv2.circle(timage, (twidth//2,twidth//2), (twidth//2)-1, (255,255,255), -1)
	tedges = auto_canny(timage).astype(bool).astype(float)
	#timage = cv2.imread('template.bmp', 0)
	#tedges = auto_canny(timage).astype(bool).astype(float)

	counter = 0

	while True:
		start = time.clock()
		
		counter += 1

		grabbed, frame = video.read()
		if len(frame.shape) == 3:
			query = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		else:
			query = frame

		if not grabbed:
			print 'failed to get next frame'
			exit(1)
		#query = cv2.imread('query.bmp', 0)
		edges = auto_canny(query).astype(bool).astype(float)

		#prepare edge map
		m = np.ones(edges.shape, dtype=float)
		out = np.where(edges>0.)
		m[out[0], out[1]] = 0.


		#prepare template
		#timage = cv2.imread('template.bmp', 0)
		#tedges = cv2.Canny(timage, 0, 255).astype(bool).astype(float)

		

		bins = 60

		#get orientation map and discretize them into bins
		tori = orientation_map(timage, tedges)
		iori = orientation_map(query, edges)
		tdori = normalize(tori, bins=bins)
		idori = normalize(iori, bins=bins)

		oriavg = []
		sampled_edges = np.zeros(iori.shape, float)
		for i in xrange(1,bins):
			#perform vectorized template average im images of the same angle
			#AngleBinTemplate/AngleBinImage
			abint = np.zeros(tori.shape, np.float32)
			abini = np.ones(iori.shape, np.float32)

			abint[np.where(tdori==i)] = 1.
			abini[np.where(idori==i)] = 0.

			if np.sum(abint) < 1:
				#print 'skipped angle bin {}'.format(i)
				continue

			res = vectorized_template_average(abini, abint, (step_size,step_size))

			oriavg.append(res)
			sampled_edges[np.where(idori==i)] = 1.

			if 20<i<30:
				cv2.imshow('result', resize(heatmap(res), width=320))
				cv2.imshow('edges-binned_edges', resize(np.hstack((edges, 1.-abini)), width=640))
				cv2.imshow('template piece', resize(abint, width=200))
				#if cv2.waitKey(0) == ord('q'):
				#	exit(0)

		#cv2.destroyAllWindows()
		a = np.array(oriavg)
		res = np.sum(a, axis=0) # + vectorized_template_average(m, tedges, (10,10))
		cv2.imshow('sampled_edges', resize(sampled_edges,  width=320))


		cv2.imshow('template', resize(heatmap(tdori), width=200))
		cv2.imshow('heatmap-edges', resize(heatmap(iori), width=320))
		cv2.imshow('result-average', resize(heatmap(res), width=320))

		#write images
		#cv2.imwrite('template.png', resize(heatmap(tdori), width=200))
		#cv2.imwrite('heatmap-edges.png', resize(heatmap(iori), width=640))
		#cv2.imwrite('result-average.png', resize(heatmap(res), width=640))

		#res = vectorized_template_average(m, tedge, (5,5))

		
		#visualize top N choices
		minval = np.sort(res, None)[:50]
		ix = np.in1d(res.ravel(), minval).reshape(res.shape)
		ay, ax = np.where(ix)

		final = cv2.cvtColor(query.copy(), cv2.COLOR_GRAY2BGR)
		for x, y in zip(ax, ay):
			x *= step_size
			y *= step_size
			final = cv2.rectangle(final, (x, y), (x+twidth,y+twidth), (0,255,0), 1)

		cv2.imshow('bounding-box', resize(final, width=320))
		#cv2.imshow('edge-map', normalize(edges, bins=255).astype(np.uint8))
		
		#cv2.imwrite('bounding-box.png', final)
		#cv2.imwrite('edge-map.png', normalize(edges, bins=255).astype(np.uint8))

		#cv2.imshow('distance_transform', normalize(distance_transform_edt(m)))
		#cv2.imshow('dist func', normalize(res))
		if cv2.waitKey(1) == ord('q'):
			exit(1)

		print '--- frame {}, {:.2}s'.format(counter, time.clock()-start)
