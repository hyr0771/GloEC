import sys
import json
import random
import torch
import os


def get_esm_data(isoform_name):
    name_list = [isoform_name]
    esm_emb = model_embedding_test(name_list, dtype=torch.float32)
    print('over')
    # torch.save(esm_emb, '../isoform/isoform_only_dataset.pt')
    return esm_emb

def model_embedding_test(name_list, dtype):
    # ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id) for id in name_list]
    esm_emb = torch.cat(esm_to_cat).to(dtype=dtype)
    return esm_emb

def load_esm(lookup):
    if os.path.exists('d:/PaperCode/DataProcess/isoform_esm_data/' + lookup + '.pt'):
        esm = format_esm(torch.load('d:/PaperCode/DataProcess/isoform_esm_data/' + lookup + '.pt'))
    else:
        print('没有找到')

    return esm.unsqueeze(0)

def format_esm(a):
    if type(a) == dict:
        a = a['mean_representations'][33]
    return a
