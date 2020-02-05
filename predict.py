import argparse
from predict_utils import PredictUtils

my_parser = argparse.ArgumentParser(description='Script used for training a model.')

my_parser.add_argument('--image_path', action='store', dest='image_path', type=str, required=True)
my_parser.add_argument('--model_path', action='store', dest='model_path', type=str, required=True)
my_parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=3)
my_parser.add_argument('--map_to_name', action='store', dest='map_to_name', type=bool, default=True)

args = my_parser.parse_args()


if __name__ == '__main__':

	util = PredictUtils()

	processed_image = util.process_image(args.image_path)

	loaded_moded = util.load_checkpoint(args.model_path)

	util.predict(process_image)
	