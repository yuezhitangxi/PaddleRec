import numpy as np
import tensorflow as tf
from collections import defaultdict

class RecDataset:
    def __init__(self, file_list, config):
        self.config = config
        self.file_list = file_list
        self.init()
        
        # 创建数据集
        self.dataset = tf.data.Dataset.from_generator(
            self._generate_data,
            output_types=self._get_output_types(),
            output_shapes=self._get_output_shapes()
        )
        
        # 可以根据需要添加预处理操作，如批处理
        batch_size = self.config.get("hyper_parameters.batch_size", 32)
        self.dataset = self.dataset.batch(batch_size)
        
        # 创建迭代器
        self.iterator = self.dataset.make_one_shot_iterator()
        self.next_batch = self.iterator.get_next()

    def init(self):
        # 定义所有字段ID
        all_field_id = [
            '101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124',
            '125', '126', '127', '128', '129', '205', '206', '207', '210',
            '216', '508', '509', '702', '853', '301'
        ]
        self.all_field_id_dict = defaultdict(int)
        self.max_len = self.config.get("hyper_parameters.max_len", 3)
        # 初始化字段ID字典，存储是否访问和索引
        for i, field_id in enumerate(all_field_id):
            self.all_field_id_dict[field_id] = [False, i]
        self.padding = 0
        # 字段数量（用于后续定义输出形状）
        self.num_fields = len(all_field_id)

    def _generate_data(self):
        """生成数据的生成器函数，供tf.data.Dataset使用"""
        for file in self.file_list:
            with open(file, "r") as rf:
                for l in rf:
                    features = l.strip().split(',')
                    ctr = int(features[1])
                    ctcvr = int(features[2])

                    # 初始化输出列表
                    output = [(field_id, []) for field_id in self.all_field_id_dict]
                    output_list = []
                    
                    # 处理特征
                    for elem in features[4:]:
                        field_id, feat_id = elem.strip().split(':')
                        if field_id not in self.all_field_id_dict:
                            continue
                        self.all_field_id_dict[field_id][0] = True
                        index = self.all_field_id_dict[field_id][1]
                        output[index][1].append(int(feat_id))

                    # 处理每个字段，确保长度一致
                    for field_id in self.all_field_id_dict:
                        visited, index = self.all_field_id_dict[field_id]
                        self.all_field_id_dict[field_id][0] = False  # 重置访问标记
                        
                        # 截断或填充到最大长度
                        if len(output[index][1]) > self.max_len:
                            processed = output[index][1][:self.max_len]
                        else:
                            processed = output[index][1] + [self.padding] * (self.max_len - len(output[index][1]))
                            
                        output_list.append(np.array(processed, dtype='int64'))
                    
                    # 添加标签
                    output_list.append(np.array([ctr], dtype='int64'))
                    output_list.append(np.array([ctcvr], dtype='int64'))
                    
                    yield tuple(output_list)  # TensorFlow的生成器需要返回元组

    def _get_output_types(self):
        """定义输出数据的类型"""
        # 字段特征都是int64，最后两个是标签
        return tuple([tf.int64] * (self.num_fields + 2))

    def _get_output_shapes(self):
        """定义输出数据的形状"""
        shapes = []
        # 每个字段特征的形状是(max_len,)
        for _ in range(self.num_fields):
            shapes.append(tf.TensorShape([self.max_len]))
        # 标签的形状是(1,)
        shapes.append(tf.TensorShape([1]))
        shapes.append(tf.TensorShape([1]))
        return tuple(shapes)

    def get_next(self):
        """获取下一个批次的数据"""
        return self.next_batch
    