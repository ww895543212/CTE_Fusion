import numpy as np
from PIL import Image
import math
import cv2
import torch

def calculate_crop_number(image, crop_height, crop_width, oc):

	height = image.shape[0]
	width = image.shape[1]
	height_number = math.ceil(height / crop_height)
	height_number = oc * (height_number - 1) + 1
	width_number = math.ceil(width / crop_width)
	width_number = oc * (width_number - 1) + 1
	output = height_number * width_number
	return output, height_number, width_number


def test_and_complement(image, crop_height, crop_width):
	if image.shape[0] != crop_height or image.shape[1] != crop_width:
		if len(image.shape) == 2:
			complement = np.zeros([crop_height, crop_width]).astype(image.dtype)  # astype数据类型转换
		else:
			complement = np.zeros([crop_height, crop_width, image.shape[2]]).astype(image.dtype)
		complement[0:image.shape[0], 0:image.shape[1]] = image
		return complement
	else:
		return image

def crop_image(image, crop_height, crop_width, oc):
	total_output_number, height_number, width_number = calculate_crop_number(image, crop_height, crop_width, oc)
	if len(image.shape) == 2:
		output = np.zeros([total_output_number, crop_height, crop_width]).astype(image.dtype)
	else:
		output = np.zeros([total_output_number, crop_height, crop_width, image.shape[2]]).astype(image.dtype)
	count = 0
	for i in range(height_number):
		for j in range(width_number):
			unit_crop_image = image[int(crop_height/oc*i):int(crop_height/oc*i)+crop_height,
						int(crop_width/oc*j):int(crop_width/oc*j)+crop_width]
			unit_crop_image = test_and_complement(unit_crop_image, crop_height, crop_width)
			output[count] = unit_crop_image
			count += 1
	return output, total_output_number
			


def img_avg(imgs):
	# f = cv2.addWeighted(imgs[0], 0.5, imgs[1], 0.5, 0)
	# tensor1 = torch.tensor(imgs[0])
	# tensor2 = torch.tensor(imgs[1])
	# f = (tensor1 + tensor2)/2
	# f = f.numpy()
	f = cv2.add(imgs[0], imgs[1]) * 0.5
	return f



def recover_image(cropped_image, height, width, crop_height, crop_width, oc):
	
	in_height_number = math.ceil(height / crop_height)
	height_number = oc * (in_height_number - 1) + 1
	in_width_number = math.ceil(width / crop_width)
	width_number = oc * (in_width_number - 1) + 1
	if len(cropped_image.shape) == 3:
		output_image = np.zeros([in_height_number*crop_height, in_width_number*crop_width]).astype(cropped_image.dtype)
	else:
		output_image = np.zeros([in_height_number*crop_height, 
				in_width_number*crop_width, cropped_image.shape[3]]).astype(cropped_image.dtype)
	assert crop_height * (oc - 1) % (2 * oc) == 0 and crop_width * (oc - 1) % (2 * oc) == 0,\
	'The input crop image size and overlap coefficient cannot meet the exact division'
	h_sec_pos = int(crop_height * (oc - 1) / (2 * oc))
	w_sec_pos = int(crop_width * (oc - 1) / (2 * oc))
	h_thi_pos = int(crop_height * (oc + 1) / (2 * oc))
	w_thi_pos = int(crop_width * (oc + 1) / (2 * oc))
	h_half_pos = int(crop_height/oc)
	w_half_pos = int(crop_width/oc)

	for i in range(height_number):
		if i == 0:   # 第一行
			for j in range(width_number):
				if height_number == 1:   # 如果就分了1行
					if j == 0:
						if width_number == 1:
							output_image[0:crop_height,0:crop_width]=\
							cropped_image[i*width_number+j][0:crop_height,0:crop_width]
						else:
							output_image[0:crop_height,0:w_thi_pos]=\
							cropped_image[i*width_number+j][0:crop_height,0:w_thi_pos]
					elif j == (width_number -1):
						output_image[0:crop_height,j*w_half_pos+w_sec_pos:] =\
						 cropped_image[i*width_number+j][0:crop_height,w_sec_pos:crop_width]
					else:
						output_image[0:crop_height,w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos] =\
						cropped_image[i*width_number+j][0:crop_height,w_sec_pos:w_thi_pos]

				else:
					if j == 0:
						if width_number == 1:
							output_image[0:h_thi_pos,0:crop_width]=\
							cropped_image[i*width_number+j][0:h_thi_pos,0:crop_width]
						else:   #  第一行第一个
							# output_image[0:h_thi_pos,0:w_thi_pos]=\
							# cropped_image[i*width_number+j][0:h_thi_pos,0:w_thi_pos]

							# avg1 = (cropped_image[i*width_number+j][0:h_half_pos,w_half_pos:h_thi_pos] +
							# 		cropped_image[i*width_number+j + 1][0:h_half_pos, 0:h_sec_pos])/2
							avg1 = img_avg([cropped_image[i*width_number+j][0:h_half_pos,w_half_pos:h_thi_pos],
									cropped_image[i*width_number+j + 1][0:h_half_pos, 0:h_sec_pos]])
							# avg2 = (cropped_image[i*width_number+j][h_half_pos:h_thi_pos, 0:w_half_pos] +
							# 		cropped_image[(i + 1)*width_number+j][0:h_sec_pos, 0:w_half_pos])/2
							avg2 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, 0:w_half_pos],
									cropped_image[(i + 1)*width_number+j][0:h_sec_pos, 0:w_half_pos]])
							# avg3 = (cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_half_pos:w_thi_pos] +
							# 		cropped_image[i*width_number+j + 1][h_half_pos:h_thi_pos, 0:w_sec_pos] +
							# 		cropped_image[(i + 1)*width_number+j][0:h_sec_pos, w_half_pos:w_thi_pos] +
							# 		cropped_image[(i + 1)*width_number+j+1][0:h_sec_pos,0:w_sec_pos])/4
							avg3_1 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_half_pos:w_thi_pos],
									cropped_image[i*width_number+j + 1][h_half_pos:h_thi_pos, 0:w_sec_pos]])
							avg3_2 = img_avg([cropped_image[(i + 1)*width_number+j][0:h_sec_pos, w_half_pos:w_thi_pos],
									cropped_image[(i + 1)*width_number+j+1][0:h_sec_pos,0:w_sec_pos]])
							avg3 = img_avg([avg3_1, avg3_2])

							output_image[0:h_half_pos, 0:w_half_pos] = cropped_image[i*width_number+j][0:h_half_pos,0:w_half_pos]
							output_image[0:h_half_pos, w_half_pos:w_thi_pos] = avg1
							output_image[h_half_pos:h_thi_pos, 0:w_half_pos] = avg2
							output_image[h_half_pos:h_thi_pos, w_half_pos:w_thi_pos] = avg3
							# s1 = Image.fromarray(output_image[0:h_thi_pos, 0:w_thi_pos])
							# s1.show()
							#
							# output_image[0:h_thi_pos, 0:w_thi_pos] = cropped_image[i*width_number+j][0:h_half_pos,0:w_half_pos]


					elif j == (width_number -1):  # 第一行最后一块
						output_image[0:h_thi_pos,j*w_half_pos+w_sec_pos:] =\
						cropped_image[i*width_number+j][0:h_thi_pos,w_sec_pos:crop_width]

						# avg1 = (cropped_image[i*width_number+j][0:h_half_pos, w_sec_pos:w_half_pos] +
						# 		cropped_image[i*width_number+j-1][0:h_half_pos, w_thi_pos:crop_width])/2
						avg1 = img_avg([cropped_image[i*width_number+j][0:h_half_pos, w_sec_pos:w_half_pos],
								cropped_image[i*width_number+j-1][0:h_half_pos, w_thi_pos:crop_width]])
						# avg2 = (cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_half_pos:crop_width] +
						# 		cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_half_pos:crop_width])/2
						avg2 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_half_pos:crop_width],
								cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_half_pos:crop_width]])
						# avg3 = (cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_sec_pos:w_half_pos] +
						# 		cropped_image[i*width_number+j-1][h_half_pos:h_thi_pos, w_thi_pos:crop_width] +
						# 		cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_sec_pos:w_half_pos] +
						# 		cropped_image[(i+1)*width_number+j-1][0:h_sec_pos, w_thi_pos:crop_width])/4
						avg3_1 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_sec_pos:w_half_pos],
								cropped_image[i*width_number+j-1][h_half_pos:h_thi_pos, w_thi_pos:crop_width]])
						avg3_2 = img_avg([cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_sec_pos:w_half_pos],
								cropped_image[(i+1)*width_number+j-1][0:h_sec_pos, w_thi_pos:crop_width]])
						avg3 = img_avg([avg3_1, avg3_2])

						output_image[0:h_half_pos, (j+1)*w_half_pos:] = cropped_image[i*width_number+j][0:h_half_pos, w_half_pos:crop_width]
						output_image[0:h_half_pos, j*w_half_pos+w_sec_pos:j*w_half_pos+w_half_pos] =avg1
						output_image[h_half_pos:h_thi_pos, (j+1)*w_half_pos:] = avg2
						output_image[h_half_pos:h_thi_pos, j*w_half_pos+w_sec_pos:j*w_half_pos+w_half_pos] = avg3

						# s1 = output_image[0:h_thi_pos,j*w_half_pos+w_sec_pos:]
						# s1 = Image.fromarray(s1)
						# s1.show()
						# s2 = 1
					else:
						# output_image[0:h_thi_pos,w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos] =\
						# cropped_image[i*width_number+j][0:h_thi_pos,w_sec_pos:w_thi_pos]

						# avg1 = (cropped_image[i*width_number+j-1][0:h_half_pos, w_thi_pos:crop_width] +
						# 		cropped_image[i*width_number+j][0:h_half_pos, w_sec_pos:w_half_pos])/2
						imgs = [cropped_image[i * width_number + j - 1][0:h_half_pos, w_thi_pos:crop_width],
						 cropped_image[i * width_number + j][0:h_half_pos, w_sec_pos:w_half_pos]]
						avg1 = img_avg([cropped_image[i*width_number+j-1][0:h_half_pos, w_thi_pos:crop_width],
								cropped_image[i*width_number+j][0:h_half_pos, w_sec_pos:w_half_pos]])
						# avg2 = (cropped_image[i*width_number+j][0:h_half_pos, w_half_pos:w_thi_pos] +
						# 		cropped_image[i*width_number+j+1][0:h_half_pos, 0: w_sec_pos])/2
						avg2 = img_avg([cropped_image[i*width_number+j][0:h_half_pos, w_half_pos:w_thi_pos],
								cropped_image[i*width_number+j+1][0:h_half_pos, 0: w_sec_pos]])
						# avg3 = (cropped_image[i*width_number+j-1][h_half_pos:h_thi_pos, w_thi_pos:crop_width] +
						# 		cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_sec_pos:w_half_pos] +
						# 		cropped_image[(i+1)*width_number+j-1][0:h_sec_pos, w_thi_pos:crop_width] +
						# 		cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_sec_pos:w_half_pos])/4
						avg3_1 = img_avg([cropped_image[i*width_number+j-1][h_half_pos:h_thi_pos, w_thi_pos:crop_width],
								cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_sec_pos:w_half_pos]])
						avg3_2 = img_avg([cropped_image[(i+1)*width_number+j-1][0:h_sec_pos, w_thi_pos:crop_width],
								cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_sec_pos:w_half_pos]])
						avg3 = img_avg([avg3_1, avg3_2])
						# avg4 = (cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_half_pos:w_thi_pos] +
						# 		cropped_image[i*width_number+j+1][h_half_pos:h_thi_pos, 0:w_sec_pos] +
						# 		cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_half_pos:w_thi_pos] +
						# 		cropped_image[(i+1)*width_number+j+1][0:h_sec_pos, 0:w_sec_pos])/4
						avg4_1 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_half_pos:w_thi_pos],
								cropped_image[i*width_number+j+1][h_half_pos:h_thi_pos, 0:w_sec_pos]])
						avg4_2 = img_avg([cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_half_pos:w_thi_pos],
								cropped_image[(i+1)*width_number+j+1][0:h_sec_pos, 0:w_sec_pos]])
						avg4 = img_avg([avg4_1, avg4_2])

						output_image[0:h_half_pos, w_thi_pos + (j - 1) * w_half_pos:w_thi_pos + (j - 1) * w_half_pos + w_sec_pos] = avg1
						output_image[0:h_half_pos, w_thi_pos + (j - 1) * w_half_pos + w_sec_pos:w_thi_pos + (j - 1) * w_half_pos + w_half_pos] = avg2
						output_image[h_half_pos:h_thi_pos, w_thi_pos + (j - 1) * w_half_pos:w_thi_pos + (j - 1) * w_half_pos + w_sec_pos] = avg3
						output_image[h_half_pos:h_thi_pos, w_thi_pos + (j - 1) * w_half_pos + w_sec_pos:w_thi_pos + (j - 1) * w_half_pos + w_half_pos] = avg4

						# s1 = output_image[0:h_thi_pos,w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos]
						# s1 = Image.fromarray(s1)
						# s1.show()
						# s2 = s1

		elif i == (height_number - 1):  # 最后一行
			for j in range(width_number):
				if j == 0:
					if width_number == 1:
						output_image[i*h_half_pos+h_sec_pos:,0:crop_width]=\
						cropped_image[i*width_number+j][h_sec_pos:crop_height,0:crop_width]
					else:
						# output_image[i*h_half_pos+h_sec_pos:,0:w_thi_pos]=\
						# cropped_image[i*width_number+j][h_sec_pos:crop_height,0:w_thi_pos]

						# avg1 = (cropped_image[i*width_number+j][h_sec_pos:h_half_pos, 0:w_half_pos] +
						# 		cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, 0:w_half_pos])/2
						avg1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, 0:w_half_pos],
								cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, 0:w_half_pos]])
						# avg2 = (cropped_image[i*width_number+j][h_half_pos:crop_height, w_half_pos:w_thi_pos] +
						# 		cropped_image[i*width_number+j+1][h_half_pos:crop_height, 0:w_sec_pos])/2
						avg2 = img_avg([cropped_image[i*width_number+j][h_half_pos:crop_height, w_half_pos:w_thi_pos],
								cropped_image[i*width_number+j+1][h_half_pos:crop_height, 0:w_sec_pos]])
						# avg3 = (cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_half_pos:w_thi_pos] +
						# 		cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_half_pos:w_thi_pos] +
						# 		cropped_image[i*width_number+j+1][h_sec_pos:h_half_pos, 0:w_sec_pos] +
						# 		cropped_image[(i-1)*width_number+j+1][h_thi_pos:crop_height, 0:w_sec_pos])/4
						avg3_1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_half_pos:w_thi_pos],
								cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_half_pos:w_thi_pos]])
						avg3_2 = img_avg([cropped_image[i*width_number+j+1][h_sec_pos:h_half_pos, 0:w_sec_pos],
								cropped_image[(i-1)*width_number+j+1][h_thi_pos:crop_height, 0:w_sec_pos]])
						avg3 = img_avg([avg3_1, avg3_2])

						output_image[i * h_half_pos + h_half_pos:, 0:w_half_pos] =\
						cropped_image[i*width_number+j][h_half_pos:crop_height, 0:w_half_pos]
						output_image[i * h_half_pos + h_sec_pos:i * h_half_pos + h_half_pos, 0:w_half_pos] = avg1
						output_image[i * h_half_pos + h_sec_pos:i * h_half_pos + h_half_pos, w_half_pos:w_thi_pos] = avg3
						output_image[i * h_half_pos + h_half_pos:, w_half_pos:w_thi_pos] = avg2

						# s1 = output_image[i * h_half_pos + h_sec_pos:, 0:w_thi_pos]
						# s1 = Image.fromarray(s1)
						# s1.show()
						# s2 = s1

				elif j == (width_number - 1):
					output_image[i*h_half_pos+h_sec_pos:,j*w_half_pos+w_sec_pos:] =\
					cropped_image[i*width_number+j][h_sec_pos:crop_height,w_sec_pos:crop_width]

					# avg1 = (cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_half_pos:] +
					# 		cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_half_pos:])
					avg1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_half_pos:],
							cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_half_pos:]])
					# avg2 = (cropped_image[i*width_number+j][h_half_pos:, w_sec_pos:w_half_pos] +
					# 		cropped_image[i*width_number+j-1][h_half_pos:, w_thi_pos:crop_width])/2
					avg2 = img_avg([cropped_image[i*width_number+j][h_half_pos:, w_sec_pos:w_half_pos],
							cropped_image[i*width_number+j-1][h_half_pos:, w_thi_pos:crop_width]])
					# avg3 = (cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_sec_pos:w_half_pos] +
					# 		cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_sec_pos:w_half_pos] +
					# 		cropped_image[i*width_number+j-1][h_sec_pos:h_half_pos, w_thi_pos:crop_width] +
					# 		cropped_image[(i-1)*width_number+j-1][h_thi_pos:crop_height, w_thi_pos:crop_width])/4
					avg3_1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_sec_pos:w_half_pos],
							cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_sec_pos:w_half_pos]])
					avg3_2 = img_avg([cropped_image[i*width_number+j-1][h_sec_pos:h_half_pos, w_thi_pos:crop_width],
							cropped_image[(i-1)*width_number+j-1][h_thi_pos:crop_height, w_thi_pos:crop_width]])
					avg3 = img_avg([avg3_1, avg3_2])

					# output_image[i * h_half_pos + h_half_pos:, j * w_half_pos + w_half_pos:] =\
					# 	cropped_image[i*width_number+j][h_half_pos:, w_half_pos:]
					output_image[i * h_half_pos + h_half_pos:, j * w_half_pos + w_half_pos:] =\
					img_avg([cropped_image[i*width_number+j][h_half_pos:, w_half_pos:],
							 cropped_image[i*width_number+j][h_half_pos:, w_half_pos:]])

					output_image[i * h_half_pos + h_half_pos:,\
					j * w_half_pos + w_sec_pos:j * w_half_pos + w_half_pos] = avg2

					output_image[i * h_half_pos + h_sec_pos:i * h_half_pos + h_half_pos,\
					j * w_half_pos + w_sec_pos:j * w_half_pos + w_half_pos] = avg3

					output_image[i * h_half_pos + h_sec_pos:i * h_half_pos + h_half_pos,\
					j * w_half_pos + w_half_pos:] = avg1

					# s1 = output_image[i*h_half_pos+h_sec_pos:,j*w_half_pos+w_sec_pos:]
					#
					# s1 = Image.fromarray(s1)
					# s1.show()
					# s2 = s1


				else:
					# output_image[i*h_half_pos+h_sec_pos:,w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos] =\
					# cropped_image[i*width_number+j][h_sec_pos:crop_height,w_sec_pos:w_thi_pos]

					# avg1 = (cropped_image[i*width_number+j][h_half_pos:crop_height, w_sec_pos:w_half_pos] +
					# 		cropped_image[i*width_number+j-1][h_half_pos:crop_height, w_thi_pos:crop_width])/2
					avg1 = img_avg([cropped_image[i*width_number+j][h_half_pos:crop_height, w_sec_pos:w_half_pos],
									cropped_image[i*width_number+j-1][h_half_pos:crop_height, w_thi_pos:crop_width]])
					# avg2 = (cropped_image[i*width_number+j][h_half_pos:crop_height, w_half_pos:w_thi_pos] +
					# 		cropped_image[i*width_number+j+1][h_half_pos:crop_height, 0:w_sec_pos])/2
					avg2 = img_avg([cropped_image[i*width_number+j][h_half_pos:crop_height, w_half_pos:w_thi_pos],
							cropped_image[i*width_number+j+1][h_half_pos:crop_height, 0:w_sec_pos]])
					# avg3 = (cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_sec_pos:w_half_pos] +
					# 		cropped_image[i*width_number+j-1][h_sec_pos:h_half_pos, w_thi_pos:crop_width] +
					# 		cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_sec_pos:w_half_pos] +
					# 		cropped_image[(i-1)*width_number+j-1][h_thi_pos:crop_height, w_thi_pos:crop_width])/4
					avg3_1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_sec_pos:w_half_pos],
							cropped_image[i*width_number+j-1][h_sec_pos:h_half_pos, w_thi_pos:crop_width]])
					avg3_2 = img_avg([cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_sec_pos:w_half_pos],
							cropped_image[(i-1)*width_number+j-1][h_thi_pos:crop_height, w_thi_pos:crop_width]])
					avg3 = img_avg([avg3_1, avg3_2])
					# avg4 = (cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_half_pos:w_thi_pos] +
					# 		cropped_image[i*width_number+j+1][h_sec_pos:h_half_pos, 0:w_sec_pos] +
					# 		cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_half_pos:w_thi_pos] +
					# 		cropped_image[(i-1)*width_number+j+1][h_thi_pos:crop_height, 0:w_sec_pos])/4
					avg4_1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_half_pos:w_thi_pos],
							cropped_image[i*width_number+j+1][h_sec_pos:h_half_pos, 0:w_sec_pos]])
					avg4_2 = img_avg([cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_half_pos:w_thi_pos],
							cropped_image[(i-1)*width_number+j+1][h_thi_pos:crop_height, 0:w_sec_pos]])
					avg4 = img_avg([avg4_1, avg4_2])

					output_image[i * h_half_pos + h_half_pos:,\
					w_thi_pos+(j-1)*w_half_pos : w_thi_pos+(j-1)*w_half_pos + w_sec_pos] = avg1

					output_image[i * h_half_pos + h_half_pos:,\
					w_thi_pos+(j-1)*w_half_pos + w_sec_pos:w_thi_pos+(j-1)*w_half_pos + w_half_pos] = avg2

					output_image[i * h_half_pos + h_sec_pos:i * h_half_pos + h_half_pos, \
					w_thi_pos + (j - 1) * w_half_pos: w_thi_pos + (j - 1) * w_half_pos + w_sec_pos] = avg3

					output_image[i * h_half_pos + h_sec_pos:i * h_half_pos + h_half_pos, \
					w_thi_pos+(j-1)*w_half_pos + w_sec_pos:w_thi_pos+(j-1)*w_half_pos + w_half_pos] = avg4

					# s1 = output_image[i*h_half_pos+h_sec_pos:,w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos]
					# s1 = Image.fromarray(s1)
					# s1.show()
					# s2 = s1

		else:
			for j in range(width_number):
				if j == 0:
					if width_number == 1:
						output_image[h_thi_pos+(i-1)*h_half_pos:h_thi_pos+i*h_half_pos,
						0:crop_width]=cropped_image[i*width_number+j][h_sec_pos:h_thi_pos,0:crop_width]
					else:   # 其他行第一个
						# output_image[h_thi_pos+(i-1)*h_half_pos:h_thi_pos+i*h_half_pos,
						# 0:w_thi_pos]=cropped_image[i*width_number+j][h_sec_pos:h_thi_pos,0:w_thi_pos]

						# avg1 = (cropped_image[i*width_number+j][h_sec_pos:h_half_pos, 0:w_half_pos] +
						# 		cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, 0:w_half_pos])/2
						avg1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, 0:w_half_pos],
										cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, 0:w_half_pos]])
						# avg2 = (cropped_image[i*width_number+j][h_half_pos:h_thi_pos, 0:w_half_pos] +
						# 		cropped_image[(i+1)*width_number+j][0:h_sec_pos, 0:w_half_pos])/2
						avg2 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, 0:w_half_pos],
										cropped_image[(i+1)*width_number+j][0:h_sec_pos, 0:w_half_pos]])
						# avg3 = (cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_half_pos:w_thi_pos] +
						# 		cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_half_pos:w_thi_pos] +
						# 		cropped_image[i*width_number+j+1][h_sec_pos:h_half_pos, 0:w_sec_pos] +
						# 		cropped_image[(i-1)*width_number+j+1][h_thi_pos:crop_height, 0:w_sec_pos])/4
						avg3_1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_half_pos:w_thi_pos],
										  cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_half_pos:w_thi_pos]])
						avg3_2 = img_avg([cropped_image[i*width_number+j+1][h_sec_pos:h_half_pos, 0:w_sec_pos],
										  cropped_image[(i-1)*width_number+j+1][h_thi_pos:crop_height, 0:w_sec_pos]])
						avg3 = img_avg([avg3_1, avg3_2])
						# avg4 = (cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_half_pos:w_thi_pos] +
						# 		cropped_image[i*width_number+j+1][h_half_pos:h_thi_pos, 0:w_sec_pos] +
						# 		cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_half_pos:w_thi_pos] +
						# 		cropped_image[(i+1)*width_number+j+1][0:h_sec_pos, 0:w_sec_pos])/4
						avg4_1 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_half_pos:w_thi_pos],
										  cropped_image[i*width_number+j+1][h_half_pos:h_thi_pos, 0:w_sec_pos]])
						avg4_2 = img_avg([cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_half_pos:w_thi_pos],
										  cropped_image[(i+1)*width_number+j+1][0:h_sec_pos, 0:w_sec_pos]])
						avg4 = img_avg([avg4_1, avg4_2])

						output_image[h_thi_pos + (i - 1) * h_half_pos:h_thi_pos + (i - 1) * h_half_pos + h_sec_pos,
						0:w_half_pos] = avg1

						output_image[h_thi_pos + (i - 1) * h_half_pos + h_sec_pos:h_thi_pos + (i - 1) * h_half_pos + h_half_pos,
						0:w_half_pos] = avg2

						output_image[h_thi_pos + (i - 1) * h_half_pos:h_thi_pos + (i - 1) * h_half_pos + h_sec_pos,
						w_half_pos:w_thi_pos] = avg3

						output_image[h_thi_pos + (i - 1) * h_half_pos + h_sec_pos:h_thi_pos + (i - 1) * h_half_pos + h_half_pos,
						w_half_pos:w_thi_pos] = avg4

						# s1 = cropped_image[i*width_number+j][h_half_pos:h_thi_pos, 0:w_half_pos]
						# s1 = Image.fromarray(s1)
						# s1.show()
						# s2 = s1

				elif j == (width_number - 1):
					output_image[h_thi_pos+(i-1)*h_half_pos:h_thi_pos+i*h_half_pos,j*w_half_pos+w_sec_pos:] =\
					cropped_image[i*width_number+j][h_sec_pos:h_thi_pos,w_sec_pos:crop_width]

					avg1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_half_pos:crop_width],
									cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_half_pos:crop_width]])
					avg2 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_half_pos:crop_width],
									cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_half_pos:crop_width]])
					avg3_1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_sec_pos:w_half_pos],
									  cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_sec_pos:w_half_pos]])
					avg3_2 = img_avg([cropped_image[i*width_number+j-1][h_sec_pos:h_half_pos, w_thi_pos:crop_width],
									  cropped_image[(i-1)*width_number+j-1][h_thi_pos:crop_height, w_thi_pos:crop_width]])
					avg3 = img_avg([avg3_1, avg3_2])
					avg4_1 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_sec_pos:w_half_pos],
									  cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_sec_pos:w_half_pos]])
					avg4_2 = img_avg([cropped_image[i*width_number+j-1][h_half_pos:h_thi_pos, w_thi_pos:crop_width],
									  cropped_image[(i+1)*width_number+j-1][0:h_sec_pos, w_thi_pos:crop_width]])
					avg4 = img_avg([avg4_1, avg4_2])

					output_image[h_thi_pos + (i - 1) * h_half_pos:h_thi_pos + (i - 1) * h_half_pos + h_sec_pos,
					j * w_half_pos + w_half_pos:] = avg1

					output_image[h_thi_pos + (i - 1) * h_half_pos + h_sec_pos:h_thi_pos + (i - 1) * h_half_pos + h_half_pos,
					j * w_half_pos + w_half_pos:] = avg2

					output_image[h_thi_pos + (i - 1) * h_half_pos:h_thi_pos + (i - 1) * h_half_pos + h_sec_pos,
					j * w_half_pos + w_sec_pos:j * w_half_pos + w_half_pos] = avg3

					output_image[h_thi_pos + (i - 1) * h_half_pos + h_sec_pos:h_thi_pos + (i - 1) * h_half_pos + h_half_pos,
					j * w_half_pos + w_sec_pos:j * w_half_pos + w_half_pos] = avg4

					# s1 = output_image[h_thi_pos+(i-1)*h_half_pos:h_thi_pos+i*h_half_pos,j*w_half_pos+w_sec_pos:]
					# s1 = Image.fromarray(s1)
					# s1.show()
					# s2 = s1


				else:
					# output_image[h_thi_pos+(i-1)*h_half_pos:h_thi_pos+i*h_half_pos,
					# w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos] = \
					# cropped_image[i*width_number+j][h_sec_pos:h_thi_pos,w_sec_pos:w_thi_pos]

					avg1_1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_sec_pos:w_half_pos],
									  cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_sec_pos:w_half_pos]])
					avg1_2 = img_avg([cropped_image[i*width_number+j-1][h_sec_pos:h_half_pos, w_thi_pos:crop_width],
									  cropped_image[(i-1)*width_number+j-1][h_thi_pos:crop_height, w_thi_pos:crop_width]])
					avg1 = img_avg([avg1_1, avg1_2])
					avg2_1 = img_avg([cropped_image[i*width_number+j][h_sec_pos:h_half_pos, w_half_pos:w_thi_pos],
									  cropped_image[(i-1)*width_number+j][h_thi_pos:crop_height, w_half_pos:w_thi_pos]])
					avg2_2 = img_avg([cropped_image[i*width_number+j+1][h_sec_pos:h_half_pos, 0:w_sec_pos],
									  cropped_image[(i-1)*width_number+j+1][h_thi_pos:crop_height, 0:w_sec_pos]])
					avg2 = img_avg([avg2_1, avg2_2])
					avg3_1 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_sec_pos:w_half_pos],
									  cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_sec_pos:w_half_pos]])
					avg3_2 = img_avg([cropped_image[i*width_number+j-1][h_half_pos:h_thi_pos, w_thi_pos:crop_width],
									  cropped_image[(i+1)*width_number+j-1][0:h_sec_pos, w_thi_pos:crop_width]])
					avg3 = img_avg([avg3_1, avg3_2])
					avg4_1 = img_avg([cropped_image[i*width_number+j][h_half_pos:h_thi_pos, w_half_pos:w_thi_pos],
									  cropped_image[(i+1)*width_number+j][0:h_sec_pos, w_half_pos:w_thi_pos]])
					avg4_2 = img_avg([cropped_image[i*width_number+j+1][h_half_pos:h_thi_pos, 0:w_sec_pos],
									  cropped_image[(i+1)*width_number+j+1][0:h_sec_pos, 0:w_sec_pos]])
					avg4 = img_avg([avg4_1, avg4_2])

					output_image[h_thi_pos + (i - 1) * h_half_pos:h_thi_pos + (i - 1) * h_half_pos + h_sec_pos,
					w_thi_pos + (j - 1) * w_half_pos:w_thi_pos + (j - 1) * w_half_pos + w_sec_pos] = avg1

					output_image[h_thi_pos + (i - 1) * h_half_pos:h_thi_pos + (i - 1) * h_half_pos + h_sec_pos,
					w_thi_pos + (j - 1) * w_half_pos + w_sec_pos:w_thi_pos + (j - 1) * w_half_pos + w_half_pos] = avg2

					output_image[h_thi_pos + (i - 1) * h_half_pos + h_sec_pos:h_thi_pos + (i - 1) * h_half_pos + h_half_pos,
					w_thi_pos + (j - 1) * w_half_pos:w_thi_pos + (j - 1) * w_half_pos + w_sec_pos] = avg3

					output_image[h_thi_pos + (i - 1) * h_half_pos + h_sec_pos:h_thi_pos + (i - 1) * h_half_pos + h_half_pos,
					w_thi_pos + (j - 1) * w_half_pos + w_sec_pos:w_thi_pos + (j - 1) * w_half_pos + w_half_pos] = avg4

					# s1 = output_image[h_thi_pos+(i-1)*h_half_pos:h_thi_pos+i*h_half_pos,
					# w_thi_pos+(j-1)*w_half_pos:w_thi_pos+j*w_half_pos]
					# s1 = Image.fromarray(s1)
					# s1.show()
					# s2 = s1


	output_image = output_image[0:height,0:width]
	return output_image
