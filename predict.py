import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.model import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from decom_test import load_image, save_image
import cv2
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

if __name__ == '__main__':
    # Clear the testA directory
    testA_dir = "./test_dataset/testA"
    for file in os.listdir(testA_dir):
        file_path = os.path.join(testA_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    os.system('python decom_test.py')

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    print(len(dataset))
    for i, data in enumerate(dataset):
        model.set_input(data)
        visuals = model.predict()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)
    webpage.save()

    
    # Combine enhanced illumination and reflectance images
    reflectance_dir = "./use_decom_dataset/cache"
    illumination_dir = "./ablation/enlightening/test_200/images"
    result_dir = "./use_decom_dataset/result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    reflectance_files = [f for f in os.listdir(reflectance_dir) if os.path.isfile(os.path.join(reflectance_dir, f))]
    for reflectance_file in reflectance_files:
        reflectance_path = os.path.join(reflectance_dir, reflectance_file)
        illumination_path = os.path.join(illumination_dir, reflectance_file.replace("reflectance_", "illumination_").replace(".png", "_fake_B.png"))
        reflectance_image = load_image(reflectance_path)
        illumination_image = load_image(illumination_path)
        
        result_image = reflectance_image * illumination_image
        save_image(result_image, os.path.join(result_dir, reflectance_file.replace("reflectance_", "result_")))

