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

def get_identity_loss(real_X, gen_YX, identity_criterion):
    '''
    Return the identity loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        identity_criterion: the identity loss function; takes the real images from X and
                        those images put through a Y->X generator and returns the identity 
                        loss (which you aim to minimize)
    '''
    
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(real_X, identity_X)
    
    return identity_loss, identity_X\

def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Return the cycle consistency loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        fake_Y: the generated images of class Y
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
                        those images put through a X->Y generator and then Y->X generator
                        and returns the cycle consistency loss (which you aim to minimize)
    '''
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(real_X, cycle_X)

    return cycle_loss, cycle_X

def get_total_gen_loss_cycle_gan(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, 
                                 identity_criterion, cycle_criterion, lambda_identity=0.1, 
                                 lambda_cycle=10):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        real_A: the real images from pile A
        real_B: the real images from pile B
        gen_AB: the generator for class A to B; takes images and returns the images 
            transformed to class B
        gen_BA: the generator for class B to A; takes images and returns the images 
            transformed to class A
        disc_A: the discriminator for class A; takes images and returns real/fake class A
            prediction matrices
        disc_B: the discriminator for class B; takes images and returns real/fake class B
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the true labels and returns a adversarial 
            loss (which you aim to minimize)
        identity_criterion: the reconstruction loss function used for identity loss
            and cycle consistency loss; takes two sets of images and returns
            their pixel differences (which you aim to minimize)
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
            those images put through a X->Y generator and then Y->X generator
            and returns the cycle consistency loss (which you aim to minimize).
            Note that in practice, cycle_criterion == identity_criterion == L1 loss
        lambda_identity: the weight of the identity loss
        lambda_cycle: the weight of the cycle-consistency loss
    '''
    adv_loss_AB, fake_B = get_gen_adversarial_loss_cycle_gan(real_A, disc_A, gen_AB, adv_criterion)    
    adv_loss_BA, fake_A = get_gen_adversarial_loss_cycle_gan(real_B, disc_B, gen_BA, adv_criterion)
    
    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
    identity_loss_AB, _ = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_BA, _ = get_identity_loss(real_B, gen_AB, identity_criterion)
    
    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    cycle_consistency_loss_AB, _ = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_consistency_loss_BA, _ = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    
    # Total loss
    gen_loss = adv_loss_AB + adv_loss_BA + \
               lambda_identity*identity_loss_AB + lambda_identity*identity_loss_BA + \
               lambda_cycle*cycle_consistency_loss_AB + lambda_cycle*cycle_consistency_loss_BA
    return gen_loss, fake_A, fake_B