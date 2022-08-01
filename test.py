# -*- coding:utf-8 -*-


import os
import torch
from torch.autograd import Variable
from net_pro import CSF_autoencoder
import utils
from args_fusion import args
import numpy as np
from crop_recover import *


def load_model(path):
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]

	csf_model = CSF_autoencoder(nb_filter, input_nc, output_nc)
	csf_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in csf_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(csf_model._get_name(), para * type_size / 1000 / 1000))

	csf_model.eval()
	csf_model.cuda()

	return csf_model


def run_demo(csf_model, infrared_path, visible_path, output_path_root, index, f_type):
	oc = 2
	img_ir, h, w, c = utils.get_test_image(infrared_path, oc=oc)
	img_vi, h, w, c = utils.get_test_image(visible_path, oc=oc)
	base_size = 224


	# dim = img_ir.shape
	if c is 1:
		if args.cuda:
			img_ir = img_ir.cuda()
			img_vi = img_vi.cuda()
		img_ir = Variable(img_ir, requires_grad=False)
		img_vi = Variable(img_vi, requires_grad=False)
		# encoder
		en_r1, en_r, en_r2 = csf_model.encoder(img_ir)
		en_v1, en_v, en_v2 = csf_model.encoder(img_vi)
		# fusion
		f = csf_model.fusion(en_r, en_r2, en_v, en_v2, f_type)
		# decoder
		recovered_image = csf_model.decoder_eval(f)[0]
		recovered_image = utils.save_image_test(recovered_image)
		recovered_image = Image.fromarray(recovered_image)
		# recovered_image.show()
		recovered_image = recovered_image.convert('L')
	else:
		# fusion each block
		img_fusion_blocks = []
		img_fusion_blocks = np.zeros([c, base_size, base_size]).astype(float)
		# img_fusion_blocks = np.zeros([c, 224, 224, 3]).astype(float)
		count = 0
		for i in range(c):
			# encoder
			img_vi_temp = torch.tensor(img_vi[i:i+1, :, :, :])
			img_ir_temp = torch.tensor(img_ir[i:i+1, :, :, :])

			img_vi_temp = img_vi_temp.float()
			img_ir_temp = img_ir_temp.float()

			if args.cuda:
				img_vi_temp = img_vi_temp.cuda()
				img_ir_temp = img_ir_temp.cuda()
			img_vi_temp = Variable(img_vi_temp, requires_grad=False)
			img_ir_temp = Variable(img_ir_temp, requires_grad=False)



			en_r1, en_r, en_r2 = csf_model.encoder(img_ir_temp)
			en_v1, en_v, en_v2 = csf_model.encoder(img_vi_temp)
			# fusion
			f = csf_model.fusion(en_r, en_r2, en_v, en_v2, f_type)
			# decoder
			img_fusion_temp = csf_model.decoder_eval(f)[0]


			img_fusion_temp = img_fusion_temp.byte()
			img_fusion_temp = img_fusion_temp.view(base_size, base_size)
			img_fusion_temp = img_fusion_temp.cpu()
			img_fusion_blocks[count] = img_fusion_temp
			count += 1

		recovered_image = recover_image(img_fusion_blocks, h, w, base_size,
										base_size, oc)


		# recovered_image.show()
		recovered_image = recovered_image.reshape([1, 1, recovered_image.shape[0], recovered_image.shape[1]])
		recovered_image = torch.tensor(recovered_image).cuda()
		recovered_image = utils.save_image_test(recovered_image)
		recovered_image = Image.fromarray(recovered_image)
		# recovered_image.show()
		recovered_image = recovered_image.convert('L')
		# recovered_image.save('./test/5.bmp')

			# img_fusion_blocks.append(img_fusion_temp)
		a = img_fusion_blocks
		# img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

	############################ multi outputs ##############################################
	output_count = 0
	# for img_fusion in img_fusion_list:
	file_name = str(index) + '.bmp'
	output_path = output_path_root + file_name
	output_count += 1
	# save images
	# utils.save_image_test(recovered_image, output_path)
	recovered_image.save(output_path)
	print(output_path)


def main():
	# run demo
	ir_root_path = r'dataset/21/ir/'
	vis_root_path = r'dataset/21/vis/'
	ct_root_path = r'dataset/medical/ct/'
	mri_root_path = r'dataset/medical/mri/'
	fusion_type = ['attention_avg', 'attention_max', 'attention_nuclear']

	with torch.no_grad():
		model_path = args.model_default
		model = load_model(model_path)
		for j in range(3):
			output_path = './outputs/21/' + fusion_type[j]

			if os.path.exists(output_path) is False:
				os.mkdir(output_path)
			output_path = output_path + '/'

			f_type = fusion_type[j]
			print('Processing......  ' + f_type)

			for i in range(21):
				index = i + 1
				infrared_path = ir_root_path + str(index) + '.bmp'
				visible_path = vis_root_path + str(index) + '.bmp'
				run_demo(model, infrared_path, visible_path, output_path, index, f_type)
	print('Done......')


if __name__ == '__main__':
	main()
