import json
import metrics


class Example(object):

    def __init__(self, file_idx, gcs, pcs, top_m_units=None, top_antecedents_index=None, top_antecedents_score=None):
        self.file_idx = file_idx
        self.gcs = [tuple(tuple([m[0], m[1]]) for m in gc) for gc in gcs]
        self.pcs = [tuple(tuple([m[0], m[1]]) for m in pc) for pc in pcs]
        self.top_m_units = top_m_units
        self.top_antecedents_index = top_antecedents_index
        self.top_antecedents_score = top_antecedents_score
        self.mention_to_gold = self.mention_to_dict(self.gcs)
        self.mention_to_predicted = self.mention_to_dict(self.pcs)

    def get_all(self):
        return self.gcs, self.mention_to_gold, self.pcs, self.mention_to_predicted

    def mention_to_dict(self, clusters):
        d = {}
        for c in clusters:
            for mention in c:
                d[tuple(mention)] = c
        return d



def read_results(file_name):
    examples = list()
    with open(file_name, "r", encoding="utf-8") as fd:
        for line in fd:
            example = json.loads(line.strip())
            file_idx = example["file_idx"]
            gold_clusters = example["gc"]
            predicted_clusters = example["pc"]
            if "top_m_units" in example:
                top_m_units = example["top_m_units"]
                top_antecedents_index = example["top_antecedents_index"]
                top_antecedents_score = example["top_antecedents_score"]
                examples.append(Example(file_idx, gold_clusters, predicted_clusters, top_m_units, top_antecedents_index, top_antecedents_score))
            else:
                examples.append(Example(file_idx, gold_clusters, predicted_clusters))
    return examples


def remove_zp_clusters(clusters):

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

def remove_zp_examples(examples):

    new_examples = list()
    for example in examples:
        file_idx, gcs, pcs = example.file_idx, example.gcs, example.pcs
        new_examples.append(Example(file_idx, remove_zp_clusters(gcs), remove_zp_clusters(pcs), example.top_m_units, example.top_antecedents_index, example.top_antecedents_score))

    return new_examples


def coref_f1(examples):
    
    coref_evaluator = metrics.CorefEvaluator()

    
    for example in examples:
        gold_clusters, mention_to_gold, predicted_clusters, mention_to_predicted = example.get_all()
        coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)

    print("muc:", coref_evaluator.evaluators[0].get_prf())
    print("bcubed:", coref_evaluator.evaluators[1].get_prf())
    print("ceaf:", coref_evaluator.evaluators[2].get_prf())
    print("average:", coref_evaluator.get_prf())



def mention_detection_f1(examples):

    true_num = 0
    gold_num = 0
    predicted_num = 0

    for example in examples:
        _, m2g, _, m2p = example.get_all()
        gold_num += len(m2g.keys())
        predicted_num += len(m2p.keys())
        true_num += len(m2g.keys() & m2p.keys())

    p = true_num / predicted_num
    r = true_num / gold_num
    f = (2 * p * r) / (p + r)

    print("p: %f, r: %f, f: %f" % (p, r, f))


def is_zp(unit):
    if unit[0] == unit[1]:
        return True
    return False


def get_predicted_clusters(top_m_units, predicted_antecedents):
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
            union_cluster = idx_to_clusters[predicted_index] | idx_to_clusters[i]
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

    return predicted_clusters


def remove_zp_and_link_next_noun(examples):
    # 重新根据先行词排序结果生成预测集，新的结果不链接的零指代，只链接第一个分数最高的非零指代
    # 返回 examples，评价指标同 coref_f1，其中 gcs 也去掉了零指代
    new_examples = list()
    for example in examples:
        file_idx = example.file_idx
        gcs = example.gcs
        top_m_units = example.top_m_units
        top_antecedents_index = example.top_antecedents_index
        top_antecedents_score = example.top_antecedents_score

        predicted_antecedents = list()
        for idx, u in enumerate(top_m_units):
            if u[0] == u[1]:
                predicted_antecedents.append(-1)
                continue
            max_score = 0
            max_antecedent_idx = -1
            for antecedent_idx, score in enumerate(top_antecedents_score[idx]):
                if score > max_score and antecedent_idx != len(top_antecedents_score[idx]) - 1 and not is_zp(top_m_units[top_antecedents_index[idx][antecedent_idx]]):
                    max_score = score
                    max_antecedent_idx = antecedent_idx
            if max_antecedent_idx != -1:
                predicted_antecedents.append(top_antecedents_index[idx][max_antecedent_idx])
            else:
                predicted_antecedents.append(-1)

        pcs = get_predicted_clusters(top_m_units, predicted_antecedents)
        new_examples.append(Example(file_idx, remove_zp_clusters(gcs), pcs))

    return new_examples



def azp_evaluation(examples):
    # 对azp作评估
    gold_azps_num = 0
    gold_zps_num = 0
    predicted_azps_num = 0
    detect_hit_azps_num = 0
    coref_hit_azps_num = 0

    for example in examples:
        gcs = example.gcs
        top_m_units = example.top_m_units
        top_antecedents_index = example.top_antecedents_index
        top_antecedents_score = example.top_antecedents_score
        example_predicted_azps = list()
        example_gold_azps = list()
        predicted_antecedents = dict()

        # 处理gcs，去掉其中的非azp
        for cluster in gcs:
            sorted_cluster = sorted(cluster, key=lambda x: x[0])
            is_azp = False
            for start, end in sorted_cluster:
                if start == end:
                    gold_zps_num += 1
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
                    if score > max_score and antecedent_idx != len(top_antecedents_score[idx]) - 1 and not is_zp(top_m_units[top_antecedents_index[idx][antecedent_idx]]):
                        max_score = score
                        max_antecedent_idx = antecedent_idx
                if max_antecedent_idx != -1:
                    predicted_antecedents[idx] = top_antecedents_index[idx][max_antecedent_idx]

        
        for zp_idx in predicted_antecedents:
            predicted_antecedent = predicted_antecedents[zp_idx]
            is_azp = True
            while is_zp(top_m_units[predicted_antecedent]):
                if predicted_antecedent in predicted_antecedents:
                    predicted_antecedent = predicted_antecedents[predicted_antecedent]
                else:
                    is_azp = False
                    break
            if is_azp == True:
                example_predicted_azps.append([top_m_units[zp_idx], top_m_units[predicted_antecedent]])

        gold_azps_num += len(example_gold_azps)
        predicted_azps_num += len(example_predicted_azps)

        for predicted_azp in example_predicted_azps:
            for gold_azp in example_gold_azps:
                if predicted_azp[0][0] == gold_azp[0][0]:
                    detect_hit_azps_num += 1
                    if tuple(predicted_azp[1]) in gold_azp[1]:
                        coref_hit_azps_num += 1
                    break


    detect_p, detect_r, detect_f = 0, 0, 0
    coref_p, coref_r, coref_f = 0, 0, 0

    detect_p = detect_hit_azps_num / predicted_azps_num
    detect_r = detect_hit_azps_num / gold_azps_num
    detect_f = 2 * detect_p * detect_r / (detect_p + detect_r)

    coref_p = coref_hit_azps_num / predicted_azps_num
    coref_r = coref_hit_azps_num / gold_azps_num
    coref_f = 2 * coref_p * coref_r / (coref_p + coref_r)
    print("gold azp nums:", gold_azps_num)
    print("predict azp nums:", predicted_azps_num)
    # print("gold zp nums:", gold_zps_num)
    print("detect: p: %f, r: %f, f: %f" % (detect_p, detect_r, detect_f))
    print("coref: p: %f, r: %f, f: %f" % (coref_p, coref_r, coref_f))



def test_wzp(file_name, flag=True):
    print("============================")
    print(file_name)
    examples = read_results(file_name)
    # 零指代相关评测
    print("-------azp f1--------------")
    azp_evaluation(examples)

    # 移除 predicted 中的零指代
    examples = remove_zp_examples(examples)
    print("-------coref f1-----------")
    coref_f1(examples)
    print("-------meniton detection f1--------")
    mention_detection_f1(examples)

    # 移除 链接层 中的零指代
    examples = remove_zp_and_link_next_noun(examples)
    print("-----移除链接层中的零指代------")
    print("-------coref f1-----------")
    coref_f1(examples)
    print("")




def test_wozp(file_name):
    print("============================")
    print(file_name)
    examples = read_results(file_name)
    print("-------coref f1-----------")
    coref_f1(examples)
    print("-------meniton detection f1--------")
    mention_detection_f1(examples)
    print("")



if __name__ == "__main__":
    
    test_wzp("./results.txt")
    exit()


