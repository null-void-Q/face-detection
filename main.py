from keras.models import load_model
import cv2
import numpy as np

def main():
	m = load_model('yolov2_tiny-face.h5')
	img = cv2.imread('face.jpeg')
	out = m.predict(preprocess(img))[0]
	print(out) 
	print(interpret_output_yolov2(out,400,320))

def preprocess(frame):
	frame =  cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame,(416,416))
	frame = frame/ 255.0
	frame= np.expand_dims(frame, axis=0)
	return frame   

def interpret_output_yolov2(output, img_width, img_height):
	anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

	netout=output
	nb_class=1
	obj_threshold=0.4
	nms_threshold=0.3

	grid_h, grid_w, nb_box = netout.shape[:3]

	size = 4 + nb_class + 1
	nb_box=5

	netout=netout.reshape(grid_h,grid_w,nb_box,size)

	boxes = []

	# decode the output by the network
	netout[..., 4]  = _sigmoid(netout[..., 4])
	netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
	netout[..., 5:] *= netout[..., 5:] > obj_threshold

	for row in range(grid_h):
		for col in range(grid_w):
			for b in range(nb_box):
				# from 4th element onwards are confidence and class classes
				classes = netout[row,col,b,5:]
				
				if np.sum(classes) > 0:
					# first 4 elements are x, y, w, and h
					x, y, w, h = netout[row,col,b,:4]

					x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
					y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
					w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
					h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
					confidence = netout[row,col,b,4]
					
					box = bounding_box(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
					
					boxes.append(box)

	# suppress non-maximal boxes
	for c in range(nb_class):
		sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

	for i in range(len(sorted_indices)):
		index_i = sorted_indices[i]
		
		if boxes[index_i].classes[c] == 0: 
			continue
		else:
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]
				
				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
					boxes[index_j].classes[c] = 0
					
	# remove the boxes which are less likely than a obj_threshold
	boxes = [box for box in boxes if box.get_score() > obj_threshold]

	result = []
	for i in range(len(boxes)):
		if(boxes[i].classes[0]==0):
			continue
		predicted_class = "face"
		score = boxes[i].score
		result.append([predicted_class,(boxes[i].xmax+boxes[i].xmin)*img_width/2,(boxes[i].ymax+boxes[i].ymin)*img_height/2,(boxes[i].xmax-boxes[i].xmin)*img_width,(boxes[i].ymax-boxes[i].ymin)*img_height,score])
	return result


if __name__ == "__main__":
	main()