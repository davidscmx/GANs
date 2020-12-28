import torch
from torch import nn

def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    gen_loss = -torch.mean(crit_fake_pred)
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred)  + c_lambda*gp
    return crit_loss

### Cycle GAN Losses ###
def get_disc_loss_cycle_gan(real_X, fake_X, disc_X, adv_criterion):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X; takes images and returns real/fake class X
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the target labels and returns a adversarial 
            loss (which you aim to minimize). Usually adv_criterion = nn.MSELoss() 
    '''
    
    out_fake = disc_X(fake_X)
    out_real = disc_X(real_X)        
    disc_fake_loss = adv_criterion(out_fake, torch.zeros_like(out_fake))
    disc_real_loss = adv_criterion(out_real, torch.ones_like(out_real))    
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss

def get_gen_adversarial_loss_cycle_gan(real_X, disc_Y, gen_XY, adv_criterion):
    '''
    Return the adversarial loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
            prediction matrices
        gen_XY: the generator for class X to Y; takes images and returns the images 
            transformed to class Y
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the target labels and returns a adversarial 
                  loss (which you aim to minimize). Usually adv_criterion = nn.MSELoss() 
    '''
    
    disc_real = disc_Y(real_X)    
    fake_Y = gen_XY(real_X)   
    disc_fake_Y = disc_Y(fake_Y)
        
    adversarial_loss = adv_criterion(disc_fake_Y, torch.ones_like(disc_fake_Y))
    
    return adversarial_loss, fake_Y