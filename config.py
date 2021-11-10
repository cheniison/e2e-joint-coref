
# best config
best_config = {
    "embedding_dim": 768,
    "max_span_width": 20,
    # max training sentences depends on size of memery 
    "max_training_sentences": 11,
    # max seq length
    "max_seq_length": 128,
    "bert_max_seq_length": 512,

    "device": "cuda",
    "checkpoint_path": "./data/checkpoint",
    "lr": 0.0002,
    "weight_decay": 0.0005,
    "dropout": 0.3,

    "report_frequency": 5,
    "eval_frequency": 40,

    # ontonotes dir
    "ontonotes_root_dir": "./data/ontonotes",
    "train_file_path": "./data/train.json",
    "test_file_path": "./data/test.json",
    "val_file_path": "./data/val.json",

    # max candidate mentions size in first/second stage
    "top_unit_ratio": 0.5,
    "max_top_antecedents": 50,
    # use coarse to fine pruning
    "coarse_to_fine": True,
    # high order coref depth
    "coref_depth": 2,

    # FFNN config
    "ffnn_depth": 1,
    "ffnn_size": 3000,

    # use span features, such as distance
    "use_features": True,
    "feature_dim": 20,
    "model_heads": True,
    # use metadata, such as genre and speaker info
    "use_metadata": True,
    "genres": ["bc", "bn", "mz", "nw", "tc", "wb"],

    # 选择topk时是否考虑单元互斥
    "extract_units": True,

    # 指代检测损失权重
    "unit_detection_loss_weight": 0.2,

    # interaction among units
    "use_units_interaction_before_score": True,        # 计算得分前的交互
    "use_units_interaction_after_score" : True,         # 计算得分后的交互
    "interaction_method_after_score": "max",           # 计算得分后的交互方式max/mean

    # 带权self attention
    "wsa_depth": 1,
    "wsa_layer_num": 1,
    "wsa_dropout": 0.3,
    "wsa_pwff_size": 3072,
    "wsa_head_num": 8,
    "wsa_lr": 0.0002,


    # transformer model
    "transformer_model_name": './data/bert-base-chinese',
    "transformer_lr": 0.00001,
}
