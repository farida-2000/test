

# import pandas as pd
# from PIL import Image
# from torchvision.transforms import ToTensor
# import torch
# import torch.nn as nn
# from video_swin_transformer import SwinTransformer3D
# from transformers import BertTokenizer, BertModel
# from torchvision import transforms
# '''
# initialize a SwinTransformer3D model for visual embeddings
# '''
# model = SwinTransformer3D()
# model.cuda()
# #print(model)
#
# #init
# # img = Image.open("C1007_9999.jpg")
# # image = ToTensor()(img).unsqueeze(0)
# # dummy_x = image[:, :, :, :, None]
# dummy_x = torch.rand(1, 3, 15,224, 224)
# logits = model(dummy_x.cuda())
# print(logits.shape)
#
# from mmcv import Config, DictAction
# from mmaction.models import build_model
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint
#
# config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
# checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'
#
# cfg = Config.fromfile(config)
# model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
# model=model.cuda()
# load_checkpoint(model, checkpoint, map_location='cpu')
#
# '''
# use the pretrained SwinTransformer3D as feature extractor
# '''
#
# # [batch_size, channel, temporal_dim, height, width]
# def vision_embedding(directory):
#     import os
#
#     for filename in os.listdir(directory):
#         f = os.path.join(directory, filename)
#         # checking if it is a file
#         if os.path.isfile(f):
#             #print(f)
#             img = Image.open(f)
#             transform = transforms.Resize((224, 224))
#             out = transform(img)
#             #print("the formated size is :",out.size)
#             image = ToTensor()(out).unsqueeze(0)
#             #print(image.shape)
#             dummy_x = image[:, :, None,:, :]
#             #print(dummy_x.shape)
#             #dummy_x = torch.rand(1, 3, 15,224, 224)
#
#             # SwinTransformer3D without cls_head
#             backbone = model.backbone
#
#             # [batch_size, hidden_dim, temporal_dim/2, height/32, width/32]
#             feat = backbone(dummy_x.cuda())
#
#             # alternative way
#             feat = model.extract_feat(dummy_x.cuda())
#
#             # mean pooling
#             feat = feat.mean(dim=[2,3,4]) # [batch_size, hidden_dim]
#
#             # project
#             batch_size, hidden_dim = feat.shape
#             feat_dim = 512
#             proj = nn.Parameter(torch.randn(hidden_dim, feat_dim))
#
#         # final output
#     output = feat.cuda() @ proj.cuda() # [batch_size, feat_dim]
#     print("vis emb of this subject is ",output) #vision embeddings based on swin video
#     return output
#
# def word_embedding_with_tokens(directory):
#     import os
#     for filename in os.listdir(directory):
#         f = os.path.join(directory, filename)
#         # checking if it is a file
#         if os.path.isfile(f):
#             my_file = open(f, "r")
#             data = my_file.read()
#             print(data)
#             # replacing end splitting the text
#             # when newline ('\n') is seen.
#             data_into_list = data.split("\n")
#             print(data_into_list)
#             my_file.close()
#             sentences=data_into_list
#             #print(data_into_list)
#             #embeddings = lang_model.encode(sentences)
#             for item in sentences:
#                 tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#                 marked_text = "[CLS] " + item + " [SEP]"
#                 tokenized_text = tokenizer.tokenize(marked_text)
#                 indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#                 segments_ids = [1] * len(tokenized_text)
#                 tokens_tensor = torch.tensor([indexed_tokens])
#                 segments_tensors = torch.tensor([segments_ids])
#                 model = BertModel.from_pretrained('bert-base-uncased',
#                                                   output_hidden_states=True,  # Whether the model returns all hidden-states.
#                                                   )
#                 model.cuda()
#                 with torch.no_grad():
#                     outputs = model(tokens_tensor.cuda(), segments_tensors.cuda())
#                     hidden_states = outputs[2]
#                 token_embeddings = torch.stack(hidden_states, dim=0)
#                 token_embeddings.size()
#                 token_embeddings = torch.squeeze(token_embeddings, dim=1)
#     print(token_embeddings)
#     print(token_embeddings.shape)
#     return token_embeddings
#
#
# #generate embeddings for subjects from data on server
# import os
# craft_hobbies_sub=["C1007","C1018","C1019","C1033","C1034","C1039","C1046","C1054","C1070","C1083","C1103","C1104","C1113","C1124","C1127","C1133","C1156","C1159","C1162","C1166","C1168","C1170","C1179","C1184","C1202","C1229","C1238","C2040","C2046"]
# df_trans=pd.read_csv('/media/data1/farida1/t/craft-hobbies/craft_fold0.csv')
# dfy=pd.read_csv('/media/data1/farida1/t/craft-hobbies/subjects_basic_info.csv')
# for k in range(1):
#     test_sub_y = df_trans[df_trans.fold == k].iloc[:, 0].tolist()
#     df_test_y = dfy[dfy['subject_id'].isin(test_sub_y)]
#     train_sub_y = df_trans[df_trans.fold != k].iloc[:, 0].tolist()
#     df_train_y = dfy[dfy['subject_id'].isin(train_sub_y)]
#     # print(df_test_y)
#     # print(df_train_y)
#     test_sub_x = df_trans[df_trans.fold == k].iloc[:, 0].tolist()
#     df_test_x = (test_sub_x)
#     train_sub_x = df_trans[df_trans.fold != k].iloc[:, 0].tolist()
#     df_train_x = (train_sub_x)
#     for i in range(len(df_train_x)):
#         directory = "/media/data1/jian/I-CONECT/crafts_hobbies/%s"% df_train_x[i]
#         print(vision_embedding(directory))
#     # for i in range(len(df_train_x)):
#     #     directory = "/media/data1/farida1/t/craft-hobbies/%s"% df_train_x[i]
#     #     print(word_embedding_with_tokens(directory))
#


import torch
import torch.nn as nn
from video_swin_transformer import SwinTransformer3D
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import BertTokenizer, BertModel
#lang_model = SentenceTransformer('bert-base-uncased')
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
model = SwinTransformer3D()

#print(model)

# image = Image.open("C1007_9998.jpg")
#
#
# transform = transforms.Compose([
#                     transforms.PILToTensor()
#                 ])
#
#
# img_tensor = transform(image)
# c=img_tensor.unsqueeze(-4)
# dummy_x=c.unsqueeze(3)
# dummy_x=dummy_x.type('torch.FloatTensor')
#
# print("image size: ")
# print(dummy_x.shape)
#
# print(dummy_x.shape)
dummy_x = torch.rand(1, 3, 32,224, 224)
logits = model(dummy_x)
print(logits.shape)
from transformers import AutoImageProcessor, SwinModel
from datasets import load_dataset
# from mmcv import Config, DictAction
# from mmaction.models import build_model
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint
#
#
# config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
# checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'
#
# cfg = Config.fromfile(config)
# model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
#
# load_checkpoint(model, checkpoint, map_location='cpu')
def word_embedding_with_tokens():

    my_file = open("C1018p.txt", "r")
    data = my_file.read()
    print(data)
    # replacing end splitting the text
    # when newline ('\n') is seen.
    data_into_list = data.split("\n")
    print(data_into_list)
    my_file.close()
    sentences=data_into_list
    #print(data_into_list)
    #embeddings = lang_model.encode(sentences)
    for item in sentences:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        marked_text = "[CLS] " + item + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        model = BertModel.from_pretrained('bert-base-uncased',
                                          output_hidden_states=True,  # Whether the model returns all hidden-states.
                                          )
        model.cuda()
        with torch.no_grad():
            outputs = model(tokens_tensor.cuda(), segments_tensors.cuda())
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings.size()
        token_embeddings = torch.squeeze(token_embeddings, dim=1)


    print(token_embeddings)
    print(token_embeddings.shape)
    return token_embeddings

a=word_embedding_with_tokens()
print(a.size)
a=a.unsqueeze(0).unfold(2, 2, 2)[0].unfold(2, 2, 2).contiguous().view(2, 6, 2, 2)
print(a.size)
