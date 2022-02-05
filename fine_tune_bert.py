import os
from sre_parse import Tokenizer
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
        # load tokenizer
        self.__load_tokenizer()

    def __load_tokenizer(self):
        # Set up tokenizer to generate Tensorflow dataset
        self.tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(self.gs_folder_bert, "vocab.txt"),do_lower_case=True)
        print("Vocab size:", len(self.tokenizer.vocab))

    def __run_tokenizer(self,text_str):
        if not self.tokenizer:
            return None
        tokens =  self.tokenizer.tokenize(text_str)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def run_inference(self, input_arr):
        result = []
        for i in input_arr:
            i_token = self.__run_tokenizer(i)
            result.append(i_token)
        
        return result



if __name__ == "__main__":
    bert_trainer = BertTrainer()
    input_arr = ['this is a test','i love my country','I hate <mask>','i hate germany']
    print(bert_trainer.run_inference(input_arr=input_arr))
    print('ran')