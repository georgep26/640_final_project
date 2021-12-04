# Planning

## Overview
* take images and articles, predict the frame

## Data
* images
* headline
* body
* url
* google visual api
* auto generated caption of lead image
* lead 3 sentences
* auto generated summary using presum

## Questions
* Based on the project description... Do we HAVE to use images?
  * sounds like yes
* Do we HAVE to use raw images in the model? Can we use the google visual 
    api column as a proxy for using the image?
* Do we HAVE to use text in conjunction with image data?

## Steps
1. Data exploration notebook
2. Preprocessing Etc.
   * Lives in github directory
   * Output datasets into data directory 
   * keep everything config driven so we know what processing goes into which dataset
   * if you request that dataset from your analysis and it doesnt exist you should use
     the config to create it 
   * config will be a json we read in, the key value will be the 
     title of the transform and the value will be a dictionary of arguments. 
     in python we say something like
```angular2html
if "tokenize" in config.keys():
    pipe.append(TokenizeTransformer(**config["tokenize"]))
```
   * do not put datasets on github (they are big) just put configs! gitignore .csv .pkl etc.
   * example data dir below
```angular2html
     * data
        |
        --> dataset_name
            |
            --> Config that created this dataset... Everything is config driven
                which transform is used on data etc
            --> dataset file (csv, parquet, pickle etc)
        --> dataset_name2
            ...
```
3. Modeling approaches
   * We try multiple modeling approaches, as they require different preprocessing we make additions to the 
     preprocessing pipeline and make a new configuration 



## Justin and George Notes
* image data
  * lets decide what pretrained model we want etc
  * outline what work needs to be done here 
  * assign things to do
  * pre trained model, transfer learn classification on classes - do this alone as baseline
  * or input to multimodal net that looks at text too
* text models
  * BERT based 
  * TFIDF etc
* Multimodal
  * throw it all into 1 model

* plan
  * Justin creating raw image classification model transfer learn from some architecture 
  * George creating raw text model from headline using BERT (then can extend to other text based features)
  * Next step after image and text unimodeal - figure out how to multimodal them 



