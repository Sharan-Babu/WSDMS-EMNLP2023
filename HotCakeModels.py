"""
WSDMS Model Implementation (EMNLP 2023)

This file implements the core WSDMS model architecture described in:
'WSDMS: Debunk Fake News via Weakly Supervised Detection of Misinforming Sentences'

The model uses a novel approach that:
1. Identifies misinforming sentences using weak supervision
2. Leverages social context through sentence-tree linking
3. Combines sentence-level predictions for article-level classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel


class HotCakeForSequenceClassification(BertPreTrainedModel):
    """
    WSDMS model for fake news detection.
    
    Key components from the paper:
    1. Input Embedding: Uses BERT for contextualized sentence representations
    2. Sentence-Tree Linking: Implemented through attention mechanisms
    3. Misinforming Sentence Detection: Graph attention for sentence analysis
    4. Article Veracity Prediction: Weighted collective attention
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
    
    def forward(self, inputs, aux_info=None):
        outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )
        
        pooled_output = outputs[1]  # Use [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Return log probabilities
        return F.log_softmax(logits, dim=1)

        # initialize parameter 
        lambda_val = 0.1

        # get attention score and embedding
        select_prob, inputs_att, inputs_att_de, z_qv_z_v_all = self.sentence_level_embedding(inputs_hiddens, inputs, msk_tensor, seg_tensor, lambda_val)
        # inputs_att = torch.cat([inputs_att, inputs_att_de], -1)

        # merge 2 embeddings
        inputs_att = torch.cat([inputs_att, inputs_att_de], -1)

        # get inference feature
        inference_feature = self.proj_inference_de(inputs_att)

        # get probability for sentence-level misinformation
        class_prob = F.softmax(inference_feature, dim=2)
        prob_sen = torch.sum(select_prob * class_prob, 1)
        # prob = torch.log(prob)

        # get attention score for each updated sentence representation
        inputs_att, att_j, z_qv_z_v_all = self.coarse_level_result(inputs_hiddens, inputs, msk_tensor, seg_tensor, lambda_val)
        # get probability for coarse-level fake news
        prob_news = torch.sum(att_j, prob_sen)

        return prob_sen, prob_news
