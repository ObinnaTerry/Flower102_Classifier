import argparse
import os
from train_utils import TrainUtils


my_parser = argparse.ArgumentParser(description='Script used for training a model.')

my_parser.add_argument('--model_name', action='store', dest='model_name', type=str, default='vgg16')
my_parser.add_argument('--base_folder', action='store', dest='base_folder', type=str, default='./flower/')
my_parser.add_argument('--mode', action='store', dest='mode', type=str, default='gpu')
my_parser.add_argument('--hidden_layer', action='store', dest='hidden_layer', type=int, default=256)
my_parser.add_argument('--lr', action='store', dest='lr', type=float, default=0.001)
my_parser.add_argument('--epochs', action=store, dest='epochs', type=int, default=3)
my_parser.add_argument('--model_save_path', action='store', dest='model_save_path', default='./')
my_parser.add_argument('--model_save_name', action='store', dest='model_save_name', default='checkpoint')

args = my_parser.parse_args()


if __name__ == '__main__':

	util = TrainUtils(args.mode)

	train_loader, valid_loader, test_loader = util.data_loader()

	model, optimizer, criterion = util.create_model(args.model_name, args.hidden_layer, args.lr)

	utils.train_validate(self, optimizer, model, criterion, train_loader, valid_loader, args.epochs)

	utils.save_model(model, train_loader, optimizer, args.model_save_path, args.model_save_name)

