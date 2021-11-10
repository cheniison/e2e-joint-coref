import model
import config
import torch
import tqdm
import tools
import json
import operator
from functools import reduce
from transformers import AutoTokenizer, AutoModel


if __name__ == "__main__":

    c = config.best_config
    
    coref_model = model.CorefModel(c).eval().to(c["device"])
    tokenizer = AutoTokenizer.from_pretrained(c["transformer_model_name"])

    transformer_model = AutoModel.from_pretrained(c["checkpoint_path"] + ".transformer.max").eval().to(c["device"])
    checkpoint = torch.load(c["checkpoint_path"] + ".max", map_location=c["device"])
    coref_model.load_state_dict(checkpoint["model"])

    
    test_data = [
        [["打雷了怎么发短信安慰女朋友？", "打雷时还给她发？"], [[[10, 13], [19, 20]], [[6, 8], [21, 21]]]]
    ]



    for data_i in test_data:

        sentences_ids, sentences_masks, _ = tools.tokenize_example(data_i, tokenizer, c)
        top_antecedents_score, top_antecedents_index, top_m_units_masks, top_m_units_start, top_m_units_end = coref_model(sentences_ids, sentences_masks, transformer_model)
        predicted_antecedents = coref_model.get_predicted_antecedents(top_antecedents_index, top_antecedents_score)
        top_m_units = list()
        for i in range(len(top_m_units_start)):
            top_m_units.append([top_m_units_start[i], top_m_units_end[i]])
        predicted_clusters, _ = coref_model.get_predicted_clusters(top_m_units, predicted_antecedents)

        print("============================")
        print("tokenized context:")
        tokens = list()
        for sentence_ids in sentences_ids:
            tokens += tokenizer.convert_ids_to_tokens(sentence_ids)
        print(tokens)
        print("predicted clusters:", predicted_clusters)