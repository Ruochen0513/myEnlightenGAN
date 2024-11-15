This is my course assignment about reproducing EnlightenGAN.</br>
I deleted some unused files.
## Train
- Create a new folder *./final.dataset* in the root dir, then create folders *./final.dataset/trainA* 和 *./final.dataset/trainB*</br>
- Put the lowlight picstures into trainA, normal pictures into trainB.
- Run by input *python scripts/script.py --train*
## Test
- Create a new folder *./test.dataset* in the root dir, then create folders *./test.dataset/testA* 和 *./test.dataset/testB*</br>
- Put the lowlight picstures to be predicted into testA, put a random picture into testB
- Run by input *python scripts/script.py --predict*
