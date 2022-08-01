# -*- coding:utf-8 -*-


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net_pro import CSF_autoencoder
from args_fusion import args
import pytorch_msssim
import copy
from loss_network import LossNetwork


def main():
	original_imgs_path = utils.list_images(args.dataset)
	train_num = 80000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)

	train(original_imgs_path)


def load_from(cfs):
	pretrained_path = 'swin_transfomer/weight/swin_tiny_patch4_window7_224.pth'
	if pretrained_path is not None:
		print("pretrained_path:{}".format(pretrained_path))
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		pretrained_dict = torch.load(pretrained_path, map_location=device)
		if "model" not in pretrained_dict:
			print("---start load pretrained modle by splitting---")
			pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
			for k in list(pretrained_dict.keys()):
				if "output" in k:
					print("delete key:{}".format(k))
					del pretrained_dict[k]
			msg = cfs.load_state_dict(pretrained_dict, strict=False)
			# print(msg)
			return
		pretrained_dict = pretrained_dict['model']
		print("---start load pretrained modle of swin encoder---")

		model_dict = cfs.state_dict()
		full_dict = copy.deepcopy(pretrained_dict)
		for k, v in pretrained_dict.items():
			if "layers." in k:
				current_layer_num = 3 - int(k[7:8])
				current_k = "layers_up." + str(current_layer_num) + k[8:]
				full_dict.update({current_k: v})
		for k in list(full_dict.keys()):
			if k in model_dict:
				if full_dict[k].shape != model_dict[k].shape:
					print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
					del full_dict[k]

		msg = cfs.load_state_dict(full_dict, strict=False)

	# print(msg)
	else:
		print("none pretrain")


def train(original_imgs_path):

	batch_size = args.batch_size

	# load network model
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]

	csf_model = CSF_autoencoder(nb_filter, input_nc, output_nc)
	with torch.no_grad():
		loss_network = LossNetwork()
		loss_network.to('cuda')
	loss_network.eval()

	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		csf_model.load_state_dict(torch.load(args.resume))
	print(csf_model)
	optimizer = Adam(csf_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		csf_model.cuda()
		# load_from(csf_model)

	tbar = trange(args.epochs)
	print('Start training.....')

	Loss_pixel = []
	Loss_ssim = []
	perceptual_loss_all = []
	Loss_all = []
	count_loss = 0
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	all_perceptual_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		csf_model.train()
		count = 0
		for batch in range(batches):
			image_paths = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img = utils.get_train_images_auto(image_paths, height=args.HEIGHT, width=args.WIDTH, flag=False)
			count += 1
			optimizer.zero_grad()
			img = Variable(img, requires_grad=False)
			if args.cuda:
				img = img.cuda()
			# get fusion image
			# encoder
			en1, en, en2 = csf_model.encoder(img)
			# en = csf_model.encoder(img)
			# decoder
			outputs = csf_model.decoder_train(en1, en, en2)
			# resolution loss: between fusion image and visible image
			x = Variable(img.data.clone(), requires_grad=False)

			ssim_loss_value = 0.
			pixel_loss_value = 0.
			perceptual_loss_value = 0.
			for output in outputs:
				pixel_loss_temp = mse_loss(output, x)
				ssim_loss_temp = ssim_loss(output, x, normalize=True)
				ssim_loss_value += (1-ssim_loss_temp)
				pixel_loss_value += pixel_loss_temp
				with torch.no_grad():
					x1 = img.detach()
				features = loss_network(x1)
				features_re = loss_network(output)

				with torch.no_grad():
					f_x_vi1 = features[1].detach()
					f_x_vi2 = features[2].detach()
					f_x_ir3 = features[3].detach()
					f_x_ir4 = features[4].detach()

				perceptual_loss = mse_loss(features_re[1], f_x_vi1) + mse_loss(features_re[2], f_x_vi2) + \
								  mse_loss(features_re[3], f_x_ir3) + mse_loss(features_re[4], f_x_ir4)
				perceptual_loss_value += perceptual_loss
			ssim_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)
			perceptual_loss_value /= len(outputs)

			# total loss
			H = args.perspectual_weight
			total_loss = pixel_loss_value + args.ssim_weight * ssim_loss_value + perceptual_loss_value*H
			total_loss.backward()
			optimizer.step()

			all_ssim_loss += ssim_loss_value.item()
			all_pixel_loss += pixel_loss_value.item()
			all_perceptual_loss += perceptual_loss_value.item()
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t SSIM weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t perceptual loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), 2, e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  (args.ssim_weight * all_ssim_loss) / args.log_interval,
								  (all_perceptual_loss*H) / args.log_interval,
								  (args.ssim_weight * all_ssim_loss + all_pixel_loss + all_perceptual_loss*H) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				perceptual_loss_all.append(all_perceptual_loss / args.log_interval)
				Loss_all.append((args.ssim_weight * all_ssim_loss + all_pixel_loss + all_perceptual_loss*H) / args.log_interval)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_pixel_loss = 0.
				all_perceptual_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				csf_model.eval()
				csf_model.cpu()
				save_model_filename = args.ssim_path + '\\' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path + ".model"
				save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
				torch.save(csf_model.state_dict(), save_model_path)
				# save loss data
				# pixel loss
				loss_data_pixel = Loss_pixel
				loss_filename_path = args.save_loss_dir + args.ssim_path + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path + ".mat"
				scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = Loss_ssim
				loss_filename_path = args.save_loss_dir + args.ssim_path + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path + ".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data = Loss_all
				loss_filename_path = args.save_loss_dir + args.ssim_path + '/' + "loss_all_epoch_" + str(e) + "_iters_" + \
									 str(count) + "-" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})

				csf_model.train()
				csf_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# pixel loss
	loss_data_pixel = Loss_pixel
	loss_filename_path = args.save_loss_dir + args.ssim_path + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_pixel': loss_data_pixel})
	loss_data_ssim = Loss_ssim
	loss_filename_path = args.save_loss_dir + args.ssim_path + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
	# SSIM loss
	loss_data = Loss_all
	loss_filename_path = args.save_loss_dir + args.ssim_path + '/' + "Final_loss_all_epoch_" + str(
		args.epochs) + "_" + str(
		time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
	# save model
	csf_model.eval()
	csf_model.cpu()
	save_model_filename = args.ssim_path + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path + ".model"
	save_model_path = os.path.join(args.save_model_dir_autoencoder, save_model_filename)
	torch.save(csf_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


# def check_paths(args):
# 	try:
# 		if not os.path.exists(args.vgg_model_dir):
# 			os.makedirs(args.vgg_model_dir)
# 		if not os.path.exists(args.save_model_dir):
# 			os.makedirs(args.save_model_dir)
# 	except OSError as e:
# 		print(e)
# 		sys.exit(1)


if __name__ == "__main__":
	main()
