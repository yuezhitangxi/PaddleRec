# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
from net import ESCMLayer  # 导入TensorFlow版本的ESCMLayer


class StaticModel:
    """
    静态图模型类，实现CTR/CVR/CTCVR多任务预测，支持反事实学习(IPW/DR)
    包含模型构建、损失计算、优化器配置等核心功能
    """
    def __init__(self, config):
        """初始化模型配置和超参数"""
        self.config = config          # 配置字典
        self._cost = None             # 总损失变量
        self.auc_states = None        # AUC计算状态变量
        self._init_hyper_parameters() # 初始化超参数
        self._init_auc_states()       # 初始化AUC状态

    def _init_hyper_parameters(self):
        """从配置中解析超参数"""
        # 数据相关参数
        self.max_len = self.config.get("hyper_parameters.max_len", 3)
        
        # 损失权重参数
        self.global_w = self.config.get("hyper_parameters.global_w", 0.5)
        self.counterfactual_w = self.config.get(
            "hyper_parameters.counterfactual_w", 0.5)
        
        # 特征相关参数
        self.sparse_feature_number = self.config.get(
            "hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get(
            "hyper_parameters.sparse_feature_dim")
        self.num_field = self.config.get("hyper_parameters.num_field")
        self.feature_size = self.config.get("hyper_parameters.feature_size")
        
        # 模型结构参数
        self.ctr_fc_sizes = self.config.get("hyper_parameters.ctr_fc_sizes")
        self.cvr_fc_sizes = self.config.get("hyper_parameters.cvr_fc_sizes")
        self.expert_num = self.config.get("hyper_parameters.expert_num")
        self.expert_size = self.config.get("hyper_parameters.expert_size")
        self.tower_size = self.config.get("hyper_parameters.tower_size")
        
        # 训练相关参数
        self.learning_rate = self.config.get(
            "hyper_parameters.optimizer.learning_rate")
        self.counterfact_mode = self.config.get("runner.counterfact_mode")

    def _init_auc_states(self):
        """初始化AUC计算所需的状态变量"""
        self.auc_states = {
            'ctr': tf.contrib.metrics.streaming_auc(
                predictions=tf.constant([0.0]), 
                labels=tf.constant([0])
            )[1],
            'cvr': tf.contrib.metrics.streaming_auc(
                predictions=tf.constant([0.0]), 
                labels=tf.constant([0])
            )[1],
            'ctcvr': tf.contrib.metrics.streaming_auc(
                predictions=tf.constant([0.0]), 
                labels=tf.constant([0])
            )[1]
        }

    def create_feeds(self, is_infer=False):
        """
        创建输入占位符
        
        Args:
            is_infer: 是否为推理模式
            
        Returns:
            输入特征和标签的占位符列表
        """
        # 创建23个稀疏特征输入占位符
        sparse_input_ids = [
            tf.placeholder(
                dtype=tf.int64,
                shape=[None, self.max_len],
                name=f"field_{i}"
            ) for i in range(23)
        ]
        
        # 创建标签占位符
        label_ctr = tf.placeholder(
            dtype=tf.int64,
            shape=[None, 1],
            name="ctr"
        )
        label_cvr = tf.placeholder(
            dtype=tf.int64,
            shape=[None, 1],
            name="cvr"
        )
        
        # 组合输入列表
        inputs = sparse_input_ids + [label_ctr] + [label_cvr]
        return inputs

    def counterfact_ipw(self, loss_cvr, ctr_num, O, ctr_out_one):
        """
        反事实IPW(Inverse Propensity Weighting)损失计算
        
        Args:
            loss_cvr: 原始CVR损失
            ctr_num: CTR样本数量
            O: 观测指示变量(是否点击)
            ctr_out_one: CTR预测的正例概率
            
        Returns:
            加权后的CVR损失
        """
        # 计算倾向得分PS
        PS = tf.multiply(
            ctr_out_one, 
            tf.cast(ctr_num, dtype=tf.float32)
        )
        
        # 防止除零操作
        min_v = tf.fill(tf.shape(PS), 0.000001)
        PS = tf.maximum(PS, min_v)
        
        # 计算IPS权重
        IPS = tf.reciprocal(PS)
        batch_shape = tf.fill(tf.shape(O), 1)
        batch_size = tf.reduce_sum(tf.cast(batch_shape, dtype=tf.float32))
        
        # 截断IPS范围(工程trick)
        IPS = tf.clip_by_value(IPS, -15, 15)
        IPS = tf.multiply(IPS, batch_size)
        IPS = tf.stop_gradient(IPS)  # 停止梯度传播
        
        # 计算加权损失
        loss_cvr = tf.multiply(loss_cvr, IPS)
        loss_cvr = tf.multiply(loss_cvr, O)
        
        return tf.reduce_mean(loss_cvr)

    def counterfact_dr(self, loss_cvr, O, ctr_out_one, imp_out):
        """
        反事实DR(Doubly Robust)损失计算
        
        Args:
            loss_cvr: 原始CVR损失
            O: 观测指示变量(是否点击)
            ctr_out_one: CTR预测的正例概率
            imp_out: 干扰项预测输出
            
        Returns:
            DR修正后的CVR损失
        """
        # 计算误差项 e = loss_cvr - imp_out
        e = tf.subtract(loss_cvr, imp_out)
        
        # 防止除零操作
        min_v = tf.fill(tf.shape(ctr_out_one), 0.000001)
        ctr_out_one = tf.maximum(ctr_out_one, min_v)
        
        # 计算IPS权重
        IPS = tf.divide(tf.cast(O, dtype=tf.float32), ctr_out_one)
        IPS = tf.clip_by_value(IPS, -15, 15)  # 截断范围
        IPS = tf.stop_gradient(IPS)  # 停止梯度传播
        
        # 计算DR误差项
        loss_error_second = tf.multiply(e, IPS)
        loss_error = tf.add(imp_out, loss_error_second)
        
        # 计算DR正则项
        loss_imp = tf.square(e)
        loss_imp = tf.multiply(loss_imp, IPS)
        
        # 总DR损失
        loss_dr = tf.add(loss_error, loss_imp)
        
        return tf.reduce_mean(loss_dr)

    def net(self, inputs, is_infer=False):
        """
        构建网络计算图
        
        Args:
            inputs: 输入特征和标签列表
            is_infer: 是否为推理模式
            
        Returns:
            包含损失和评估指标的字典
        """
        # 初始化ESCMLayer模型
        escm_model = ESCMLayer(
            sparse_feature_number=self.sparse_feature_number,
            sparse_feature_dim=self.sparse_feature_dim,
            num_field=self.num_field,
            ctr_layer_sizes=self.ctr_fc_sizes,
            cvr_layer_sizes=self.cvr_fc_sizes,
            expert_num=self.expert_num,
            expert_size=self.expert_size,
            tower_size=self.tower_size,
            counterfact_mode=self.counterfact_mode,
            feature_size=self.feature_size
        )
        
        # 前向传播计算
        out_list = escm_model.forward(inputs[0:-2])
        
        # 解析模型输出
        (ctr_out, ctr_out_one, cvr_out, 
         cvr_out_one, ctcvr_prop, ctcvr_prop_one) = out_list[0:6]
        
        # 解析标签
        ctr_clk = inputs[-2]       # CTR标签(点击)
        ctcvr_buy = inputs[-1]     # CTCVR标签(购买)
        
        # 转换标签为float32
        ctr_clk_float = tf.cast(ctr_clk, dtype=tf.float32)
        ctcvr_buy_float = tf.cast(ctcvr_buy, dtype=tf.float32)
        
        # 计算辅助变量
        ctr_num = tf.reduce_sum(ctr_clk_float, axis=0)
        O = ctr_clk_float  # 观测指示变量(是否点击)

        # 计算AUC指标
        auc_ctr, update_auc_ctr = tf.contrib.metrics.streaming_auc(
            predictions=tf.slice(ctr_out, [0, 1], [-1, 1]),
            labels=ctr_clk_float,
            curve='ROC',
            stateful=True
        )
        auc_ctcvr, update_auc_ctcvr = tf.contrib.metrics.streaming_auc(
            predictions=tf.slice(ctcvr_prop, [0, 1], [-1, 1]),
            labels=ctcvr_buy_float,
            curve='ROC',
            stateful=True
        )
        auc_cvr, update_auc_cvr = tf.contrib.metrics.streaming_auc(
            predictions=tf.slice(cvr_out, [0, 1], [-1, 1]),
            labels=ctcvr_buy_float,
            curve='ROC',
            stateful=True
        )

        # 推理模式返回
        if is_infer:
            return {
                'auc_ctr': auc_ctr,
                'auc_cvr': auc_cvr,
                'auc_ctcvr': auc_ctcvr,
                'update_ops': [update_auc_ctr, update_auc_cvr, update_auc_ctcvr]
            }

        # 训练模式：计算损失
        # CTR损失
        loss_ctr = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ctr_clk_float,
            logits=tf.log(ctr_out_one / (1 - ctr_out_one))
        )
        
        # CVR损失（基础）
        loss_cvr = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ctcvr_buy_float,
            logits=tf.log(cvr_out_one / (1 - cvr_out_one))
        )
        
        # 根据反事实模式调整CVR损失
        if self.counterfact_mode == "DR":
            loss_cvr = self.counterfact_dr(loss_cvr, O, ctr_out_one, out_list[6])
        else:
            loss_cvr = self.counterfact_ipw(loss_cvr, ctr_num, O, ctr_out_one)
        
        # CTCVR损失
        loss_ctcvr = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ctcvr_buy_float,
            logits=tf.log(ctcvr_prop_one / (1 - ctcvr_prop_one))
        )

        # 总损失（加权求和）
        cost = loss_ctr + loss_cvr * self.counterfactual_w + loss_ctcvr * self.global_w
        avg_cost = tf.reduce_mean(cost)
        self._cost = avg_cost  # 保存总损失

        # 返回训练指标
        return {
            'cost': avg_cost,
            'auc_ctr': auc_ctr,
            'auc_cvr': auc_cvr,
            'auc_ctcvr': auc_ctcvr,
            'update_ops': [update_auc_ctr, update_auc_cvr, update_auc_ctcvr]
        }

    def create_optimizer(self, strategy=None):
        """
        创建优化器
        
        Args:
            strategy: 分布式策略
            
        Returns:
            训练操作
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        
        # 分布式策略支持
        if strategy is not None:
            optimizer = tf.train.SyncReplicasOptimizer(
                optimizer, replicas_to_aggregate=strategy
            )
        
        # 最小化损失
        return optimizer.minimize(self._cost)

    def infer_net(self, inputs):
        """推理模式接口"""
        return self.net(inputs, is_infer=True)
    