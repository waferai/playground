import os
import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

"""
taken from here -  https://www.tensorflow.org/text/tutorials/fine_tune_bert
"""

class BertTrainer:

    def __init__(self) -> None:
        self.gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
        tf.io.gfile.listdir(self.gs_folder_bert)
        glue, info = tfds.load('glue/mrpc', with_info=True,batch_size=-1)
        #info describe the dataset and features.
        # print(list(glue.keys()))
        # print(info.features)
        # print(info.features['label'].names)

        glue_train = glue['train']

        # for key, value in glue_train.items():
        #     print(f"{key:9s}: {value[0].numpy()}")
        
        self.__process_tokenizer()

        test_sentence = 'breastfeeding'
        print(self.tokenizer.tokenize(test_sentence))

    def __process_tokenizer(self):
        # Set up tokenizer to generate Tensorflow dataset
        self.tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(self.gs_folder_bert, "vocab.txt"),do_lower_case=True)



if __name__ == "__main__":
    bert_trainer = BertTrainer()
    print('ran')