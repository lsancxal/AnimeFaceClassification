# Anime Face Classification

The data comes from the paper AniWho: A Quick and Accurate Way to Classify Anime Character Faces in Images (https://arxiv.org/pdf/2208.11012v3). The dataset consists of 9,738 images across 130 character classes, with approximately 75 images per class, sourced from the Danbooru website—a platform developed by the Japanese animation-style cartoon community.

This project classifies anime faces to 130 classes using a convolutional neural network. 


## Run with Docker

docker compouse up --build


### Notes

- Training outputs are written to `./output`.