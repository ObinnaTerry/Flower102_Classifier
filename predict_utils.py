

class PredictUtils:

	def __init__(self):

		pass

	def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath, map_location=lambda storage, loc:storage)

    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    model.eval()

    return model

    def process_image(image):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)

    image_norm = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(244),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    return image_norm(image)

    