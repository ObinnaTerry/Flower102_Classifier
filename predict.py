import argparse
from predict_utils import PredictUtils

my_parser = argparse.ArgumentParser(description='Script used making predictions.')

my_parser.add_argument('--image_path', action='store', dest='image_path', type=str, required=True)
my_parser.add_argument('--model_path', action='store', dest='model_path', type=str, required=True)
my_parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=3)
my_parser.add_argument('--map_to_name', action='store', dest='map_to_name', type=bool, default=True)
my_parser.add_argument('--mode', action='store', dest='mode', type=str, default='cpu')
my_parser.add_argument('--json_file', action='store', dest='json_file', type=str, default=None)

args = my_parser.parse_args()


if __name__ == '__main__':

	util = PredictUtils()

	processed_image = util.process_image(args.image_path)

	loaded_moded = util.load_checkpoint(args.model_path)

	result = util.predict(processed_image, loaded_moded, args.top_k, args.json_file, args.mode)

	print('Predictions are as follow:\n---------------------')
	if args.json_file is None:
		print(result)
	else:
		for key, value in result.items():
			print(f'{key}: {round(value*100, 3)}%')
