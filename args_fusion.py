
class args():
	# training args
	epochs = 1
	batch_size = 1
	dataset = r'F:\paper_workspace\train_dataset\train2014'  # the dataset path in your computer
	HEIGHT = 224
	WIDTH = 224

	save_model_dir_autoencoder = "weights\\model"
	save_loss_dir = '.\\weights\\loss\\'

	cuda = 1
	ssim_weight = 100
	ssim_path = '1e2'
	perspectual_weight = 1000

	lr = 1e-4  #"learning rate, default is 0.001"
	lr_light = 1e-4  # "learning rate, default is 0.001"
	log_interval = 10  #"number of images after which the training loss is logged, default is 500"
	resume = None

	# for test, model_default is the model used in paper
	p100_ex = r'models/CSF.pth'
	model_default = p100_ex


