from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()


class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den



class AZPCorefEvaluator(object):
    """ Evaluator for anaphoric zero pronoun
    """
    def __init__(self):
        self.p_num = 0
        self.g_num = 0
        self.hit_num = 0
        super(AZPCorefEvaluator, self).__init__()

    def get_f1(self):
        return f1(self.hit_num, self.p_num, self.hit_num, self.g_num)

    def get_recall(self):
        return 0 if self.hit_num == 0 else self.hit_num / float(self.g_num)

    def get_precision(self):
        return 0 if self.hit_num == 0 else self.hit_num / float(self.p_num)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def update(self, gold_clusters, top_m_units, top_antecedents_index, top_antecedents_score):

        example_predicted_azps = list()
        example_gold_azps = list()
        predicted_antecedents = dict()

        # 处理gcs，去掉其中的非azp
        for cluster in gold_clusters:
            sorted_cluster = sorted(cluster, key=lambda x: x[0])
            is_azp = False
            for start, end in sorted_cluster:
                if start == end:
                    if is_azp:
                        example_gold_azps.append([(start, end), sorted_cluster])
                elif start != end:
                    is_azp = True

        # 得到预测结果中零指代指向的对象
        for idx, u in enumerate(top_m_units):
            if u[0] == u[1]:
                max_score = 0
                max_antecedent_idx = -1
                for antecedent_idx, score in enumerate(top_antecedents_score[idx]):
                    if score > max_score and antecedent_idx != len(top_antecedents_score[idx]) - 1:# and not is_zp(top_m_units[top_antecedents_index[idx][antecedent_idx]]):
                        max_score = score
                        max_antecedent_idx = antecedent_idx
                if max_antecedent_idx != -1:
                    predicted_antecedents[idx] = top_antecedents_index[idx][max_antecedent_idx].item()

        
        for zp_idx in predicted_antecedents:
            predicted_antecedent = predicted_antecedents[zp_idx]
            is_azp = True
            while top_m_units[predicted_antecedent][0] == top_m_units[predicted_antecedent][1]:
                if predicted_antecedent in predicted_antecedents:
                    predicted_antecedent = predicted_antecedents[predicted_antecedent]
                else:
                    is_azp = False
                    break
            if is_azp == True:
                example_predicted_azps.append([top_m_units[zp_idx], top_m_units[predicted_antecedent]])

        self.g_num += len(example_gold_azps)
        self.p_num += len(example_predicted_azps)

        for predicted_azp in example_predicted_azps:
            for gold_azp in example_gold_azps:
                if predicted_azp[0][0] == gold_azp[0][0]:
                    if tuple(predicted_azp[1]) in gold_azp[1]:
                        self.hit_num += 1
                    break


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = np.transpose(np.asarray(linear_sum_assignment(-scores)))

    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem

