# FOR LOCAL USE:

# if you want to run this with the embeddings you need to download and install them
# I didn't push the pretrained embeddings to the online repository because it's a large file

# to download the embeddings click
https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.3.0/en_vectors_web_lg-2.3.0.tar.gz

# in your environment, install the embeddings with this command
pip install en_vectors_web_lg-2.3.0.tar.gz

# must change the command when running train.py to change embedding dimensions
# python train.py --test_corpus full --nembed_word 300

# FOR COLAB USE:
# in colab, use Common_Grounding_Reproduction_embeds.ipynb. this already includes the commands for downloading
# and installing the embeddings

# FOR GPU POTSDAM USE:
# ?