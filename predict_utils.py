

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

