import torch
import numpy as np
from collections import OrderedDict
import metrics
import tools
import math


class CorefModel(torch.nn.Module):

    def __init__(self, config):
        """ 模型初始化函数

        parameters:
            config: 模型配置
                embedding_dim: 词向量的维度
                span_dim: span的维度
                gap_dim: 零指代维度
                ffnn_depth/size: 前馈神经网络深度和大小
                max_span_width: span的最多字符数
        """
        super(CorefModel, self).__init__()
        
        # 配置初始化

        self.config = config
        self.span_dim = 2 * self.config["embedding_dim"]
        uu_input_dim = 0
        mention_score_dim = 0

        if self.config["use_features"]:
            self.span_dim += self.config["feature_dim"]
            self.span_width_embeddings = torch.nn.Embedding(self.config["max_span_width"] + 1, self.config["feature_dim"])
            self.bucket_distance_embeddings = torch.nn.Embedding(10, self.config["feature_dim"])
            uu_input_dim += self.config["feature_dim"]

        if self.config["model_heads"]:
            self.span_dim += self.config["embedding_dim"]
            self.Sh = torch.nn.Linear(self.config["embedding_dim"], 1)      # token head score

        if self.config["use_metadata"]:
            self.genre_embeddings = torch.nn.Embedding(len(self.config["genres"]) + 1, self.config["feature_dim"])
            self.same_speaker_emb = torch.nn.Embedding(2, self.config["feature_dim"])
            uu_input_dim += 2 * self.config["feature_dim"]

        uu_input_dim += 3 * self.span_dim

        # 模型初始化

        self.gap_dim = self.config["embedding_dim"]
        zp_score_dim = 0

        self.sentence_begin_embed = torch.nn.Parameter(torch.zeros(self.config["embedding_dim"], requires_grad=True)).to(device=self.config["device"])
        # token to gap
        self.t2g = self._create_ffnn(2 * self.config["embedding_dim"], self.gap_dim, self.config["embedding_dim"], self.config["ffnn_depth"], self.config["dropout"])

        # 使用得分计算前交互
        if self.config["use_units_interaction_before_score"]:
            zp_score_dim += self.span_dim
            mention_score_dim += self.gap_dim
            self.Sspan = torch.nn.Linear(self.span_dim, 1)      # span head score
            self.Sgap = torch.nn.Linear(self.gap_dim, 1)        # gap head score

        zp_score_dim += self.gap_dim
        self.Sz = self._create_score_ffnn(zp_score_dim)            # zero pronoun score

        mention_score_dim += self.span_dim
        self.Sm = self._create_score_ffnn(mention_score_dim)     # mention score
        self.Suu = self._create_score_ffnn(uu_input_dim)      # pairwise score between units
        self.c2fP = torch.nn.Linear(self.span_dim, self.span_dim)       # coarse to fine pruning unit projection
        self.hoP = torch.nn.Linear(2 * self.span_dim, self.span_dim)    # high order projection



    def _create_ffnn(self, input_size, output_size, ffnn_size, ffnn_depth, dropout=0):
        """ 创建前馈神经网络
        """
        current_size = input_size
        model_seq = OrderedDict()
        for i in range(ffnn_depth):
            model_seq['fc' + str(i)] = torch.nn.Linear(current_size, ffnn_size)
            model_seq['relu' + str(i)] = torch.nn.ReLU(inplace=True)
            model_seq['dropout' + str(i)] = torch.nn.Dropout(dropout)
            current_size = ffnn_size
        model_seq['output'] = torch.nn.Linear(current_size, output_size)

        return torch.nn.Sequential(model_seq)


    def _create_score_ffnn(self, input_size):
        """ 创建评分前馈神经网络
        """
        return self._create_ffnn(input_size, 1, self.config["ffnn_size"], self.config["ffnn_depth"], self.config["dropout"])

    
    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        float_distances = distances.float()
        combined_idx = torch.floor(torch.log(float_distances) / math.log(2)) + 3
        use_identity = distances <= 4
        combined_idx[use_identity] = float_distances[use_identity]
        combined_idx = combined_idx.long()

        return torch.clamp(combined_idx, 0, 9)


    def get_span_embed(self, tokens_embed, spans_start, spans_end):
        """
        得到token序列中的所有span表示
        """
        span_embed_list = list()
        start_embed = tokens_embed[spans_start]           # 第一个token表示
        end_embed = tokens_embed[spans_end - 1]            # 最后一个token表示

        span_embed_list.append(start_embed)
        span_embed_list.append(end_embed)

        if self.config["model_heads"]:
            tokens_score = self.Sh(tokens_embed).view(-1)           # size: num_tokens
            tokens_locs = torch.arange(start=0, end=len(tokens_embed), dtype=torch.long).repeat(len(spans_start), 1)        # size: num_spans * num_tokens
            tokens_masks = (tokens_locs >= spans_start.view(-1, 1)) & (tokens_locs < spans_end.view(-1, 1))             # size: num_spans * num_tokens
            tokens_weights = torch.nn.functional.softmax(
                (tokens_score + torch.log(tokens_masks.float()).to(device=torch.device(self.config["device"]))), 
                dim=1
            )
            
            span_head_emb = torch.matmul(tokens_weights, tokens_embed)
            span_embed_list.append(span_head_emb)
            
        if self.config["use_features"]:
            spans_width = (spans_end - spans_start).to(device=torch.device(self.config["device"]))
            span_width_embed = self.span_width_embeddings(spans_width)
            span_width_embed = torch.nn.functional.dropout(span_width_embed, p=self.config["dropout"], training=self.training)
            span_embed_list.append(span_width_embed)

        return torch.cat(span_embed_list, dim=1)


    def get_gap_embed(self, tokens_embed):
        """ 得到位置loc处前的gap表示

        parameters:
            tokens_embed: num_sentence * max_sentence_len * embed_dim
        
        """
        num_sentence, max_sentence_len, embed_dim = tokens_embed.shape

        # num_sentence * max_sentence_len * (2 * embed_dim)
        t2g_input = torch.cat(
            (
                torch.cat((self.sentence_begin_embed.expand(num_sentence, 1, embed_dim), tokens_embed), dim=1)[:,:max_sentence_len,:],
                tokens_embed
            ),
            dim=-1
        )
        
        gaps_embed = torch.sigmoid(self.t2g(t2g_input))
        return gaps_embed


    def units_interaction_before_score(self, spans_start, spans_end, spans_embed, gaps_start, gaps_embed):
        """ 得分前交互
        """
        gaps_start_unsqueezed = gaps_start.unsqueeze(1)
        interaction_mask = (gaps_start_unsqueezed > spans_start) & (gaps_start_unsqueezed < spans_end)     # size: num_gaps * num_spans

        # get span attend embed
        span_attend_scores = self.Sspan(spans_embed)        # size: num_spans * 1
        dummy_mask = torch.zeros(interaction_mask.shape[0])
        dummy_mask[torch.sum(interaction_mask, dim=1) == 0] = 1
        # size: num_gaps * (num_spans+1)
        span_attend_weights = torch.nn.functional.softmax(
            (
                torch.cat((span_attend_scores.t(), torch.tensor([[0.]], device=self.config["device"])), dim=1) + 
                torch.log(torch.cat((interaction_mask.float(), dummy_mask.view(-1, 1)), dim=1)).to(device=torch.device(self.config["device"]))
            ), 
            dim=1
        )
        # size: num_gaps * span_dim
        span_attend_embeds = torch.matmul(span_attend_weights, torch.cat((spans_embed, torch.zeros((1, self.span_dim), device=self.config["device"])), dim=0))

        # get gap attend embed
        gap_attend_scores = self.Sgap(gaps_embed)           # size: num_gaps * 1
        dummy_mask = torch.zeros(interaction_mask.shape[1])
        dummy_mask[torch.sum(interaction_mask, dim=0) == 0] = 1
        # size: num_spans * (num_gaps + 1)
        gap_attend_weights = torch.nn.functional.softmax(
            (
                torch.cat((gap_attend_scores, torch.tensor([[0.]], device=self.config["device"])), dim=0) + 
                torch.log(torch.cat((interaction_mask.float(), dummy_mask.view(1, -1)), dim=0)).to(device=torch.device(self.config["device"]))
            ), 
            dim=0
        )
        # size: num_spans * gap_dim
        gap_attend_embeds = torch.matmul(gap_attend_weights.t(), torch.cat((gaps_embed, torch.zeros((1, self.gap_dim), device=self.config["device"])), dim=0))

        spans_score = self.Sm(torch.cat((spans_embed, gap_attend_embeds), dim=1))          # size: num_spans * 1
        gaps_score = self.Sz(torch.cat((gaps_embed, span_attend_embeds), dim=1))            # size: num_gaps * 1

        return spans_score, gaps_score

    
    def units_interaction_after_score(self, spans_start, spans_end, spans_score, gaps_start, gaps_score, interaction_method="mean"):
        """ 得分后交互
        """

        gaps_start_unsqueezed = gaps_start.unsqueeze(1)
        interaction_mask = (gaps_start_unsqueezed > spans_start) & (gaps_start_unsqueezed < spans_end)     # size: num_gaps * num_spans

        if interaction_method == "max":
            
            interaction_mask_score = torch.log(interaction_mask.float()).to(device=torch.device(self.config["device"]))

            # update spans score
            gaps_spans_score = gaps_score + interaction_mask_score              # size: num_gaps * num_spans
            gaps_max_score, _ = torch.max(gaps_spans_score, dim=0)              # size: num_spans
            gaps_max_score[gaps_max_score < 0] = 0.

            # update gaps score
            spans_gaps_score = spans_score.t() + interaction_mask_score     # size: num_gaps * num_spans
            spans_max_score, _ = torch.max(spans_gaps_score, dim=1)
            spans_max_score[spans_max_score < 0] = 0.

            spans_score -= gaps_max_score.view(-1, 1)       # update spans score
            gaps_score -= spans_max_score.view(-1, 1)       # update gaps score

        elif interaction_method == "mean":

            # update spans score
            gaps_spans_score = gaps_score.clone().repeat(1, len(spans_score))     # size: num_gaps * num_spans
            gaps_spans_score[~interaction_mask] = 0.
            gaps_valid_num = torch.sum(interaction_mask.float(), dim=0).to(device=torch.device(self.config["device"]))
            gaps_valid_num[gaps_valid_num == 0.] = 1.
            gaps_sum_score = torch.sum(gaps_spans_score, dim=0)
            gaps_mean_score = gaps_sum_score / gaps_valid_num
            gaps_mean_score[gaps_mean_score < 0] = 0.

            # update gaps score
            spans_gaps_score = spans_score.clone().t().repeat(len(gaps_score), 1)     # size: num_gaps * num_spans
            spans_gaps_score[~interaction_mask] = 0.
            spans_valid_num = torch.sum(interaction_mask.float(), dim=1).to(device=torch.device(self.config["device"]))
            spans_valid_num[spans_valid_num == 0.] = 1.
            spans_sum_score = torch.sum(spans_gaps_score, dim=1)
            spans_mean_score = spans_sum_score / spans_valid_num
            spans_mean_score[spans_mean_score < 0] = 0.

            spans_score -= gaps_mean_score.view(-1, 1)       # update spans score
            gaps_score -= spans_mean_score.view(-1, 1)       # update gaps score

        else:
            raise Exception("Unknown interaction method: %s" % interaction_method)

        return spans_score, gaps_score


    def extract_units(self, units_score, units_start, units_end, m):
        """ 得到top m个不互斥的unit
        """
        sorted_units_score, indices = torch.sort(units_score, 0, True)
        top_m_units_index = list()
        top_m_units_start = torch.zeros(m, dtype=torch.long)
        top_m_units_end = torch.zeros(m, dtype=torch.long)
        top_m_len = 0
        i = 0
        while top_m_len < m and i < len(sorted_units_score):
            unit_index = indices[i]
            unit_start = units_start[unit_index]
            unit_end = units_end[unit_index]

            res = (((unit_start < top_m_units_start) & (unit_end < top_m_units_end) & (unit_end > top_m_units_start)) 
                | ((unit_start > top_m_units_start) & (unit_start < top_m_units_end) & (unit_end > top_m_units_end))
                | ((unit_start == unit_end) & (unit_start > top_m_units_start) & (unit_end < top_m_units_end))
                | ((top_m_units_start == top_m_units_end) & (unit_start < top_m_units_start) & (unit_end > top_m_units_end)))

            if torch.sum(res) == 0:
                top_m_units_index.append(unit_index)
                top_m_units_start[top_m_len] = unit_start
                top_m_units_end[top_m_len] = unit_end
                top_m_len += 1
            i += 1

        return torch.stack(top_m_units_index)


    def get_gaps_weight(self, gaps_score, sentences_mask, gaps_in_topk):
        """ 根据gap的得分，得到每个gap的权重，并将权重组织成 sentences_mask 的形式

        parameters:
            gaps_score: num_gaps * 1
            sentences_mask: num_sentence * max_sentence_len
            gaps_in_topk: num_gaps
        return:
            gaps_weight: num_sentence * max_sentence_len
        """
        flat_gaps_weight = (gaps_score.view(-1) - max(gaps_score)) * 10        # num_gaps
        flat_gaps_weight[~gaps_in_topk] = float("-inf")

        gaps_weight = torch.full(sentences_mask.shape, float("-inf")).to(device=self.config["device"])
        gaps_weight[sentences_mask.bool()] = flat_gaps_weight
        return gaps_weight


    def coarse_to_fine_pruning(self, k, units_masks, units_embed, units_score):
        """ 由粗到精得到每个unit的候选先行语

        parameters:
            k: int, 第二阶段候选先行词个数，若配置中coarse_to_fine设置为false，此参数将被忽略
            units_masks: m * m, 候选units间的可见性
            units_embed: m * span_dim, 候选units embed
            units_score: m * 1, 候选units的当前得分

        return
            score: FloatTensor m * k
            index: Long m * k
        """

        m = len(units_embed)
        all_score = torch.zeros((m, m), device=self.config["device"])
        all_score[~units_masks] = float("-inf")

        # add unit score
        all_score += units_score

        antecedents_offset = (torch.arange(0, m).view(-1, 1) - torch.arange(0, m).view(1, -1))
        antecedents_offset = antecedents_offset.to(device=torch.device(self.config["device"]))

        if self.config["coarse_to_fine"] == True:
            source_top_unit_emb = torch.nn.functional.dropout(self.c2fP(units_embed), p=self.config["dropout"], training=self.training)
            target_top_unit_emb = torch.nn.functional.dropout(units_embed, p=self.config["dropout"], training=self.training)
            all_score += source_top_unit_emb.matmul(target_top_unit_emb.t())     # m * m
        else:
            # 使用所有候选unit
            k = m

        top_antecedents_fast_score, top_antecedents_index = torch.topk(all_score, k)
        top_antecedents_offset = torch.gather(antecedents_offset, dim=1, index=top_antecedents_index)

        return top_antecedents_fast_score, top_antecedents_index, top_antecedents_offset


    def get_units_similarity_score(self, top_antecedents_index, units_embed, top_antecedents_offset, speaker_ids, genre_emb):
        """ 得到unit间相似度得分

        parameters:
            top_antecedents_index: Long m * k, 每个unit的topk候选先行词下标
            units_embed: m * span_dim, 候选units embed
            top_antecedents_offset: m * k, 候选units间的相对偏移
            speaker_ids: m, 候选units的speaker id
            genre_emb: feature_dim, 文档类型embed
        return:
            score: FloatTensor m * k
        """
        m = len(units_embed)
        k = top_antecedents_index.shape[1]

        unit_index = torch.arange(0, m, dtype=torch.long).repeat(k, 1).t()

        uu_ffnn_input_list = list()
        uu_ffnn_input_list.append(units_embed[unit_index])
        uu_ffnn_input_list.append(units_embed[top_antecedents_index])
        uu_ffnn_input_list.append(uu_ffnn_input_list[0] * uu_ffnn_input_list[1])

        if self.config["use_features"]:
            top_antecedents_distance_bucket = self.bucket_distance(top_antecedents_offset)
            top_antecedents_distance_emb = self.bucket_distance_embeddings(top_antecedents_distance_bucket)
            uu_ffnn_input_list.append(top_antecedents_distance_emb)

        if self.config["use_metadata"]:
            same_speaker_ids = (speaker_ids.view(-1, 1) == speaker_ids[top_antecedents_index]).long().to(device=torch.device(self.config["device"]))
            speaker_emb = self.same_speaker_emb(same_speaker_ids)
            uu_ffnn_input_list.append(speaker_emb)
            uu_ffnn_input_list.append(genre_emb.repeat(m, k, 1))

        uu_ffnn_input = torch.cat(uu_ffnn_input_list, dim=2)
        uu_slow_score = self.Suu(uu_ffnn_input)

        return uu_slow_score.view(m, k)


    def forward(self, sentences_ids, sentences_masks, sentences_valid_masks, speaker_ids, sentence_map, subtoken_map, genre, transformer_model, wsa_model=None):
        """ 
        parameters:
            sentences_ids: num_sentence * max_sentence_len
            sentences_masks: num_sentence * max_sentence_len
            sentences_valid_masks: num_sentence * max_sentence_len
            speaker_ids: list[list]
            sentence_map: list[list]
            subtoken_map: list[list]
            genre: genre_id
            transformer_model: AutoModel
            wsa_model: Weighted self attention
            necessary_units: list, units won't be pruned
        """
        max_sentence_len = sentences_ids.shape[-1]

        flattened_sentence_indices = list()
        for sm in sentence_map:
            flattened_sentence_indices += sm
        flattened_sentence_indices = torch.LongTensor(flattened_sentence_indices)           # size: num_tokens

        # get units location(start/end)
        sentences_token_embed, _ = transformer_model(sentences_ids.to(device=torch.device(self.config["device"])), sentences_masks.to(device=torch.device(self.config["device"])), return_dict=False)      # size: num_sentence * max_sentence_len * embed_dim
        sentences_gap_embed = self.get_gap_embed(sentences_token_embed)            # size: num_sentence * max_sentence_len * embed_dim
        gaps_weight = None

        for _ in range(self.config["wsa_depth"]):

            # Weighted Self Attention
            if wsa_model is not None:
                
                # 拼接gap/token embed
                embeds = torch.cat((sentences_token_embed, sentences_gap_embed), dim=1)        # size: num_sentence * (2 * max_sentence_len) * embed_dim

                # 得到权重
                if gaps_weight is None:
                    gaps_weight = torch.full(sentences_gap_embed.shape[:2], float("-inf"), device=torch.device(self.config["device"]))   # size: num_sentence * max_sentence_len

                weights = torch.zeros(embeds.shape[:2], device=torch.device(self.config["device"]))            # size: num_sentence * (2 * max_sentence_len)
                weights[:,:max_sentence_len][~sentences_masks.bool()] = float("-inf")            # token weight
                weights[:,max_sentence_len:] = gaps_weight                                # gap weight

                # 输入wsa
                embeds = wsa_model(embeds, weights)

                # 分离embed
                sentences_token_embed = embeds[:,:max_sentence_len,:]
                sentences_gap_embed = embeds[:,max_sentence_len:,:]
                

            tokens_embed = sentences_token_embed[sentences_valid_masks.bool()]          # size: num_tokens * embed_dim
            gaps_embed = sentences_gap_embed[sentences_valid_masks.bool()]              # size: num_tokens(num_gaps) * embed_dim

            num_tokens = len(tokens_embed)
            
            candidate_units_start = torch.arange(0, num_tokens).repeat(self.config["max_span_width"] + 1, 1).t()
            candidate_units_end = candidate_units_start + torch.arange(0, self.config["max_span_width"] + 1)
            candidate_units_start_sentence_indices = flattened_sentence_indices[candidate_units_start]
            candidate_units_end_sentence_indices = flattened_sentence_indices[torch.min(candidate_units_end, torch.tensor(num_tokens - 1))]
            candidate_units_mask = (
                (candidate_units_end < num_tokens) & 
                (candidate_units_start_sentence_indices == candidate_units_end_sentence_indices)
            )

            units_start = candidate_units_start[candidate_units_mask]
            units_end = candidate_units_end[candidate_units_mask]

            # span start/end/embed
            spans_mask = (units_start != units_end)
            spans_start = units_start[spans_mask]
            spans_end = units_end[spans_mask]
            spans_embed = self.get_span_embed(tokens_embed, spans_start, spans_end)       # size: num_spans * span_dim
            # gap start
            gaps_mask = ~spans_mask
            gaps_start = units_start[gaps_mask]                     # size: num_gaps
            assert len(gaps_start) == len(gaps_embed)

            # units score
            if self.config["use_units_interaction_before_score"]:
                spans_score, gaps_score = self.units_interaction_before_score(spans_start, spans_end, spans_embed, gaps_start, gaps_embed)
            else:
                spans_score = self.Sm(spans_embed)
                gaps_score = self.Sz(gaps_embed)
        
            if self.config["use_units_interaction_after_score"]:
                spans_score, gaps_score = self.units_interaction_after_score(spans_start, spans_end, spans_score, gaps_start, gaps_score, self.config["interaction_method_after_score"])
                

            # integrate units
            # gap embed to unit embed
            zps_embed = torch.cat((gaps_embed, gaps_embed, gaps_embed), dim=1)
            if self.config["use_features"]:
                width_embed = self.span_width_embeddings(torch.zeros(len(zps_embed), dtype=torch.long, device=self.config["device"]))
                zps_embed = torch.cat((zps_embed, width_embed), dim=1)

            units_score = torch.empty((len(units_start), 1), device=self.config["device"])
            units_score[spans_mask] = spans_score
            units_score[gaps_mask] = gaps_score
            units_embed = torch.empty((len(units_start), self.span_dim), device=self.config["device"])
            units_embed[spans_mask] = spans_embed
            units_embed[gaps_mask] = zps_embed


            # get top m units
            units_len = len(units_score)
            units_score_mask = torch.zeros(units_len, dtype=torch.bool)
            m = min(int(self.config["top_unit_ratio"] * num_tokens), units_len)

            if self.config["extract_units"]:
                unit_indices = self.extract_units(units_score, units_start, units_end, m)
            else:
                _, unit_indices = torch.topk(units_score, m, dim=0, largest=True)

            units_score_mask[unit_indices] = True

            # Weighted Self Attention --- gaps weight update
            if wsa_model is not None:
                gaps_in_topk = units_score_mask[gaps_mask]
                gaps_weight = self.get_gaps_weight(gaps_score, sentences_valid_masks, gaps_in_topk)


        top_m_units_embed = units_embed[units_score_mask]        # size: m * span_dim
        top_m_units_score = units_score[units_score_mask]        # size: m * 1
        top_m_units_start = units_start[units_score_mask]        # size: m
        top_m_units_end = units_end[units_score_mask]            # size: m
        m = len(top_m_units_embed)

        # BoolTensor, size: m * m
        top_m_units_masks = \
            ((top_m_units_start.repeat(m, 1).t() > top_m_units_start) \
            | ((top_m_units_start.repeat(m, 1).t() == top_m_units_start) \
                & (top_m_units_end.repeat(m, 1).t() > top_m_units_end)))

        if self.config['use_metadata']:
            # use metadata
            flattened_speaker_ids = list()
            for si in speaker_ids:
                flattened_speaker_ids += si
            flattened_speaker_ids = torch.LongTensor(flattened_speaker_ids)
            top_m_units_speaker_ids = flattened_speaker_ids[top_m_units_start]
            genre_emb = self.genre_embeddings(torch.tensor([genre], dtype=torch.long, device=self.config["device"])).squeeze()
        else:
            top_m_units_speaker_ids = None
            genre_emb = None


        # coarse to fine pruning
        k = min(self.config["max_top_antecedents"], m)
        top_antecedents_fast_score, top_antecedents_index, top_antecedents_offset = self.coarse_to_fine_pruning(
            k, top_m_units_masks, top_m_units_embed, top_m_units_score
        )


        # get units similarity score
        for _ in range(self.config["coref_depth"]):
            top_antecedents_slow_score = self.get_units_similarity_score(top_antecedents_index, top_m_units_embed, top_antecedents_offset, top_m_units_speaker_ids, genre_emb)
            top_antecedents_score = top_antecedents_fast_score + top_antecedents_slow_score         # size: m * k
            dummy_score = torch.zeros((m, 1), device=self.config["device"])          # add dummy
            top_antecedents_score = torch.cat((top_antecedents_score, dummy_score), dim=1)          # size: m * (k+1)
            top_antecedents_weight = torch.nn.functional.softmax(top_antecedents_score, dim=1)      # size: m * (k+1)
            top_antecedents_emb = torch.cat((top_m_units_embed[top_antecedents_index], top_m_units_embed.unsqueeze(1)), dim=1)     # size: m * (k+1) * embed
            attended_units_emb = torch.sum(top_antecedents_weight.unsqueeze(2) * top_antecedents_emb, dim=1)            # size: m * embed
            f = torch.sigmoid(self.hoP(torch.cat([top_m_units_embed, attended_units_emb], dim=1)))  # size: m * embed
            top_m_units_embed = f * attended_units_emb + (1 - f) * top_m_units_embed                # size: m * embed


        return top_antecedents_score, top_antecedents_index, top_m_units_masks, top_m_units_start, top_m_units_end, units_score, units_start, units_end
        

    def evaluate(self, data, transformer_model, wsa_model=None):
        """ evaluation
        """
        import json
        coref_evaluator = metrics.CorefEvaluator()
        wozp_evaluator = metrics.CorefEvaluator()
        azp_evaluator = metrics.AZPCorefEvaluator()

        fd = open("./results.txt", "w")

        with torch.no_grad():
            for file_idx, data_i in enumerate(data):
                sentences_ids, sentences_masks, sentences_valid_masks, gold_clusters, speaker_ids, sentence_map, subtoken_map, genre = data_i
                if max([len(s) for s in sentences_ids]) > self.config["bert_max_seq_length"]:
                    # 忽略超过bert长度的example
                    continue
                top_antecedents_score, top_antecedents_index, top_m_units_masks, top_m_units_start, top_m_units_end, units_score, units_start, units_end = self.forward(sentences_ids, sentences_masks, sentences_valid_masks, speaker_ids, sentence_map, subtoken_map, genre, transformer_model, wsa_model)
                predicted_antecedents = self.get_predicted_antecedents(top_antecedents_index, top_antecedents_score)
                top_m_units = list()
                for i in range(len(top_m_units_start)):
                    top_m_units.append(tuple([top_m_units_start[i].item(), top_m_units_end[i].item()]))

                # all units
                gold_clusters = [tuple(tuple([m[0], m[1]]) for m in gc) for gc in gold_clusters]
                mention_to_gold = {}
                for gc in gold_clusters:
                    for mention in gc:
                        mention_to_gold[tuple(mention)] = gc
                predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_m_units, predicted_antecedents)
                coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

                azp_evaluator.update(gold_clusters, top_m_units, top_antecedents_index, top_antecedents_score)

                output = {"file_idx": file_idx, "gc": gold_clusters, "pc": predicted_clusters, "top_m_units": top_m_units, "top_antecedents_index": top_antecedents_index.tolist(), "top_antecedents_score": top_antecedents_score.tolist()}
                fd.write(json.dumps(output) + "\n")

                # units without zero pronouns
                gold_clusters = self.remove_zp_cluster(gold_clusters)
                mention_to_gold = {}
                for gc in gold_clusters:
                    for mention in gc:
                        mention_to_gold[tuple(mention)] = gc
                predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_m_units, predicted_antecedents, remove_zp=True)
                wozp_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

        fd.close()
        return coref_evaluator.get_prf(), wozp_evaluator.get_prf(), azp_evaluator.get_prf()


    def get_predicted_antecedents(self, top_antecedents_index, top_antecedents_score):
        """ 得到每个unit的得分最高的先行词
        """
        predicted_antecedents = []
        for i, index in enumerate(torch.argmax(top_antecedents_score, axis=1)):
            if index == len(top_antecedents_score[i]) - 1:
                # 空指代
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(top_antecedents_index[i][index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_m_units, predicted_antecedents, remove_zp=False):
        """ 根据预测的先行词得到指代簇
        """
        idx_to_clusters = {}
        predicted_clusters = []
        for i in range(len(top_m_units)):
            idx_to_clusters[i] = set([i])

        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            else:
                union_cluster = idx_to_clusters[predicted_index.item()] | idx_to_clusters[i]
                for j in union_cluster:
                    idx_to_clusters[j] = union_cluster
        
        tagged_index = set()
        for i in idx_to_clusters:
            if (len(idx_to_clusters[i]) == 1) or (i in tagged_index):
                continue
            cluster = idx_to_clusters[i]
            predicted_cluster = list()
            for j in cluster:
                tagged_index.add(j)
                predicted_cluster.append(tuple(top_m_units[j]))

            predicted_clusters.append(tuple(predicted_cluster))

        if remove_zp == True:
            predicted_clusters = self.remove_zp_cluster(predicted_clusters)

        mention_to_predicted = {}
        for pc in predicted_clusters:
            for mention in pc:
                mention_to_predicted[tuple(mention)] = pc

        return predicted_clusters, mention_to_predicted

    def remove_zp_cluster(self, clusters):
        # 移除聚类中的零指代，和tools中的略有不同
        clusters_wo_zp = list()
        for cluster in clusters:
            cluster_wo_zp = list()
            for sloc, eloc in cluster:
                if eloc - sloc > 0:
                    cluster_wo_zp.append(tuple([sloc, eloc]))           # 此处需强制转成 tuple
            if len(cluster_wo_zp) > 1:
                clusters_wo_zp.append(tuple(cluster_wo_zp))

        return clusters_wo_zp




class WeightedSelfAttention(torch.nn.Module):

    def __init__(self, config):
        super(WeightedSelfAttention, self).__init__()
        self.config = config
        N = self.config["wsa_layer_num"]
        dropout = self.config["wsa_dropout"]
        pwff_size = self.config["wsa_pwff_size"]
        head_num = self.config["wsa_head_num"]
        model_dim = self.config["embedding_dim"]
        self.encoder = torch.nn.ModuleList([WSASublayer(model_dim, head_num, pwff_size, dropout) for _ in range(N)])


    def forward(self, embeds, weights):
        """ weighted self attention

        parameters:
            embeds: FloatTensor, b * m * model_dim
            weights: FloatTensor, b * m
        """

        for layer in self.encoder:
            embeds = layer(embeds, weights)

        return embeds



class WSASublayer(torch.nn.Module):

    def __init__(self, model_dim, head_num, pwff_size, dropout):
        super(WSASublayer, self).__init__()

        assert model_dim % head_num == 0
        self.model_dim = model_dim
        self.head_num = head_num
        self.d_k = self.model_dim // self.head_num
        self.pwff_size = pwff_size

        self.linear_q = torch.nn.Linear(self.model_dim, self.model_dim)
        self.linear_k = torch.nn.Linear(self.model_dim, self.model_dim)
        self.linear_v = torch.nn.Linear(self.model_dim, self.model_dim)
        self.linear_multihead = torch.nn.Linear(self.model_dim, self.model_dim)

        # Position-wise Feed Forward
        self.pwff = torch.nn.Sequential(
            torch.nn.Linear(self.model_dim, self.pwff_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.pwff_size, self.model_dim)
        ) 

        self.norm1 = torch.nn.LayerNorm(self.model_dim)
        self.norm2 = torch.nn.LayerNorm(self.model_dim)
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, embeds, weights):
        """ weighted self attention

        parameters:
            embeds: FloatTensor, b * m * model_dim
            weights: FloatTensor, b * m
        """
        batch_size = embeds.shape[0]

        Q = self.linear_q(embeds).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)           # b * h * m * d_k
        K = self.linear_k(embeds).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)           # b * h * m * d_k
        V = self.linear_v(embeds).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)           # b * h * m * d_k

        output = self.weighted_dot_product_attention(Q, K, V, weights)      
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.d_k)     # b * m * model_dim
        output = self.linear_multihead(output)                              # b * m * model_dim
        output = self.norm1(embeds + self.dropout(output))
        
        x = output
        output = self.pwff(output)
        output = self.norm2(x + self.dropout(output))

        return output


    def weighted_dot_product_attention(self, Q, K, V, weights):
        """ weighted dot product attention
        
        parameters:
            Q: b * h * m * d_k, query tensor
            K: b * h * m * d_k, key tensor
            V: b * h * m * d_k, value tensor
            weights: b * m, token weight
        return:
            score: FloatTensor b * h * m * d_k
        """
        attend_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) + weights.unsqueeze(1).unsqueeze(1)          # size: b * h * m * m
        attend_weights = torch.nn.functional.softmax(attend_weights, dim=-1)
        attend_weights = self.dropout(attend_weights)

        return torch.matmul(attend_weights, V)           # size: b * h * m * d_k
