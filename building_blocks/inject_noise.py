import torch
from torch import nn

class InjectNoise(nn.Module):
    '''
    Inject Noise Class
    Values:
        channels: the number of channels the image has, a scalar
    '''
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter( 
            # Use nn.Parameter so that these weights can be optimized
            # Initiate the weights for the channels from a random normal distribution            
            torch.nn.Parameter(torch.randn(size = (1, channels, 1, 1)))
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns the image with random noise added.
        Parameters:
            image: the feature map of shape (n_samples, channels, width, height)
        '''                
        # Then, your model would create 512 values—one for each channel.                 
        n_samples, channels, width, height = image.shape
        noise_shape = (n_samples, 1, width, height)        
        noise = torch.randn(noise_shape, device=image.device) # Creates the random noise
        return image + (self.weight * noise) # Applies to image after multiplying by the weight for each channel
    
    #UNIT TEST COMMENT: Required for grading
    def get_weight(self):
        return self.weight
    
    #UNIT TEST COMMENT: Required for grading
    def get_self(self):
        return self
    