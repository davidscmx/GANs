
# Build Basic Generative Adversarial Network

### Syllabus:
Week 1: Intro to GANs
Week 2: Deep Convolutional GANs
Week 3: Wasserstein GANs with Gradient Penalty
Week 4: Conditional GAN & Controllable Generation

## Week 1: Intro to GANSs

GAN: Generative Adversarial Network

There are two networks in GAN's:
Two model: The generator and the discriminator
And they fight against each other

### Generative Models

- Can generate realistic images, music

What are generative models?

Types of generative models


Generative vs. Discriminative Models

- Discriminative models
    - Given features X what is the probability or  P(Y|X)

- Generative models
    - Given a noise, class vectors get the features P(X|Y)

There are many types of generative model:
- Variational Autoencoders

- Generative adversarial networks
    - Generator - discriminator
    - Similar to the decoder
    - But no guiding encoder
    - Models compete and learn gainst each other 
    - At some point we don't need the discriminator
    - Generative models learn to produce realistic examples 

## Real-Life GANs

- Only around since 2014
- Cool applications
    -Produce photorealistical photographies
    - Image translation -> Horse to zebra, drawing photorealistically
    - 
- Major companies
    - Adobe ->
    - Google -> Data text augmentation
    - Creative filters 

## Intuition Behind GANs
### Outline
    - The goal of the generator and the discriminator
    - The competition between them.

Generator learns to make fakes that look real - _The forger_
Discriminator learns to distiguish what is real - _the art inspector_

- Generator is not allowed to see the new images
- We tell the discriminator is if it guessed correctly



## Generator

- The heart of the GAN

- The noise vector is fed in as input in the generator's neural network. 

### Generator: Learning

Noise -> Neural Network generator-> Features(X hat) -> Discriminator-> Output Yd -> Cost


### BCE: Binary Cross Entropy Cost Function
