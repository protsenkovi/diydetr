Work in progress.

# Architecture

## Highlevel

![high level architecture](docs/images/high_level_architecture.png)

## DETR Transformer 

![detr transformer architecture](docs/images/detr_transformer_architecture.jpeg)

# Misc

## Losses comparison

intersection decrease, union increase, distance between centers increse for losses:
- intersection over union
- union over enclosure
- generalized intersection over union (https://giou.stanford.edu)
- distance + intersection over union (https://arxiv.org/abs/1911.08287)
- euclidian distance between centers over euclidian distance between top-left and bottow-right corners of the enclosure

https://user-images.githubusercontent.com/431393/169621968-430f9874-d30d-4789-9e32-53f579b90f9e.mp4

intersection and union increases, distance stay fixed

https://user-images.githubusercontent.com/431393/169621980-18a63704-d820-4b61-a797-73d61e960240.mp4


intersection over union and dod losses

https://user-images.githubusercontent.com/431393/169622000-ad89b079-9550-4eb2-8bfc-de74c8229169.mp4


## predicted to target boxes assignment 

https://user-images.githubusercontent.com/431393/169622005-27d07ab5-b357-47d7-a3c9-7696b3462c77.mp4


## visualisation of training process run for several epochs with fixed batch 

sign that the code is starting to work.

[https://raw.githubusercontent.com/protsenkovi/diydetr/master/docs/videos/example_convergence_for_fixed_batch.mp4](https://user-images.githubusercontent.com/431393/169621836-fd76bf23-6050-4404-80c4-55663c32216b.mp4)
