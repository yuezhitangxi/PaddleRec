import tensorflow as tf
import math

class ESCMLayer:
    def __init__(self, sparse_feature_number, sparse_feature_dim, num_field,
                 ctr_layer_sizes, cvr_layer_sizes, expert_num, expert_size,
                 tower_size, counterfact_mode, feature_size):
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.num_field = num_field
        self.ctr_layer_sizes = ctr_layer_sizes
        self.cvr_layer_sizes = cvr_layer_sizes
        self.counterfact_mode = counterfact_mode
        self.expert_num = expert_num
        self.expert_size = expert_size
        self.tower_size = tower_size
        self.gate_num = 3 if counterfact_mode == "DR" else 2
        self.feature_size = feature_size
        
        # 初始化嵌入层
        self.embedding_weights = tf.get_variable(
            name="SparseFeatFactors",
            shape=[self.sparse_feature_number, self.sparse_feature_dim],
            initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        )
        
        # 初始化专家网络
        self.experts = []
        for i in range(self.expert_num):
            expert = self._create_linear_layer(
                input_size=self.feature_size,
                output_size=self.expert_size,
                name=f'expert_{i}'
            )
            self.experts.append(expert)
        
        # 初始化门控网络、塔层和输出层
        self.gates = []
        self.towers = []
        self.tower_outs = []
        
        for i in range(self.gate_num):
            # 门控网络
            gate = self._create_linear_layer(
                input_size=self.feature_size,
                output_size=self.expert_num,
                name=f'gate_{i}'
            )
            self.gates.append(gate)
            
            # 塔层
            tower = self._create_linear_layer(
                input_size=self.expert_size,
                output_size=self.tower_size,
                name=f'tower_{i}'
            )
            self.towers.append(tower)
            
            # 输出层
            tower_out = self._create_linear_layer(
                input_size=self.tower_size,
                output_size=2,
                name=f'tower_out_{i}'
            )
            self.tower_outs.append(tower_out)
    
    def _create_linear_layer(self, input_size, output_size, name):
        """创建线性层，包含权重和偏置"""
        with tf.variable_scope(name, reuse=False):
            weights = tf.get_variable(
                name='weights',
                shape=[input_size, output_size],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            biases = tf.get_variable(
                name='biases',
                shape=[output_size],
                initializer=tf.constant_initializer(0.1)
            )
        return (weights, biases)
    
    def _linear_layer(self, x, weights, biases):
        """应用线性层：y = x * W + b"""
        return tf.matmul(x, weights) + biases
    
    def forward(self, inputs):
        """前向传播函数"""
        # 计算嵌入向量
        embeddings = []
        for data in inputs:
            # 获取嵌入向量
            feat_emb = tf.nn.embedding_lookup(self.embedding_weights, data)
            #  sum pooling
            feat_emb = tf.reduce_sum(feat_emb, axis=1)
            embeddings.append(feat_emb)
        
        # 拼接所有嵌入向量
        concat_emb = tf.concat(embeddings, axis=1)
        
        # 专家网络计算
        expert_outputs = []
        for i in range(self.expert_num):
            weights, biases = self.experts[i]
            linear_out = self._linear_layer(concat_emb, weights, biases)
            expert_output = tf.nn.relu(linear_out)
            expert_outputs.append(expert_output)
        
        # 拼接并重塑专家输出
        expert_concat = tf.concat(expert_outputs, axis=1)
        expert_concat = tf.reshape(
            expert_concat, 
            [-1, self.expert_num, self.expert_size]
        )
        
        # 门控机制和塔层计算
        output_layers = []
        for i in range(self.gate_num):
            # 门控计算
            gate_weights, gate_biases = self.gates[i]
            cur_gate_linear = self._linear_layer(concat_emb, gate_weights, gate_biases)
            cur_gate = tf.nn.softmax(cur_gate_linear)
            cur_gate = tf.reshape(cur_gate, [-1, self.expert_num, 1])
            
            # 结合专家输出和门控
            cur_gate_expert = tf.multiply(expert_concat, cur_gate)
            cur_gate_expert = tf.reduce_sum(cur_gate_expert, axis=1)
            
            # 塔层计算
            tower_weights, tower_biases = self.towers[i]
            cur_tower = self._linear_layer(cur_gate_expert, tower_weights, tower_biases)
            cur_tower = tf.nn.relu(cur_tower)
            
            # 输出层计算
            out_weights, out_biases = self.tower_outs[i]
            out = self._linear_layer(cur_tower, out_weights, out_biases)
            out = tf.nn.softmax(out)
            out = tf.clip_by_value(out, 1e-15, 1.0 - 1e-15)
            
            output_layers.append(out)
        
        # 计算各种输出
        ctr_out = output_layers[0]
        cvr_out = output_layers[1]
        
        # 提取正样本的概率
        ctr_prop_one = tf.slice(ctr_out, [0, 1], [-1, 1])
        cvr_prop_one = tf.slice(cvr_out, [0, 1], [-1, 1])
        
        # 计算CTR*CVR
        ctcvr_prop_one = tf.multiply(ctr_prop_one, cvr_prop_one)
        ctcvr_prop = tf.concat([1 - ctcvr_prop_one, ctcvr_prop_one], axis=1)
        
        # 构建输出列表
        out_list = [
            ctr_out,
            ctr_prop_one,
            cvr_out,
            cvr_prop_one,
            ctcvr_prop,
            ctcvr_prop_one
        ]
        
        # 如果是DR模式，添加额外的输出
        if self.counterfact_mode == "DR":
            imp_out = output_layers[2]
            imp_prop_one = tf.slice(imp_out, [0, 1], [-1, 1])
            out_list.append(imp_prop_one)
        
        return out_list
    