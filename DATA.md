# Nuscene - Official SDK

## download nuscene

the official website provides a gui interface for downloading nusence datset. [This github thread](https://github.com/nutonomy/nuscenes-devkit/issues/110) describes how to download and unzip in server environment

## Using the dataset

Nuscene provides an [official SDK](https://github.com/nutonomy/nuscenes-devkit) with tutorials, including setup and [tutorial](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_tutorial.ipynb).

## Conversion of dataset

As the origional author demands the dataset to be converted to their form, we have provided data conversion code under ```parse_data/```

### sample_data caviats
 
* when there are no previous or next img in sequence, the respetive token will be ""

# Nuscene - Author's code

## Loading in program

Author's code can be found in ```datasets/nuscenes.py``` We can see that 
