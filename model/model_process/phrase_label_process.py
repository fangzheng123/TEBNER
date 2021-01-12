# encoding: utf-8

import numpy as np
from sklearn import neighbors

class PhraseLabelProcess(object):
    """
    短语标注模型处理
    """
    def label_phrase_by_knn(self, seed_entity_dict, all_phrase_list, all_entity_vec_dict, all_phrase_vec_dict):
        """
        通过KNN打标短语类型
        :param seed_entity_dict:
        :param all_phrase_list:
        :param all_entity_vec_dict:
        :param all_phrase_vec_dict:
        :return:
        """
        entity_vec_list = []
        entity_type_list = []
        entity_name_list = []
        for entity_name, entity_vec in all_entity_vec_dict.items():
            entity_vec_list.append(entity_vec)
            entity_type_list.append(seed_entity_dict[entity_name])
            entity_name_list.append(entity_name)
        entity_type_index_dict = {entity_type: index for index, entity_type in enumerate(list(set(entity_type_list)))}
        entity_index_type_dict = {index: entity_type for entity_type, index in entity_type_index_dict.items()}
        type_index_list = [entity_type_index_dict[entity_type] for entity_type in entity_type_list]

        knn_clf = neighbors.KNeighborsClassifier(3, weights="uniform", metric="euclidean")\
            .fit(np.array(entity_vec_list), np.array(type_index_list))

        # 挖掘短语类型
        phrase_entity_dict = {}
        for phrase in all_phrase_list:
            # if len(phrase) < 4:
            #     continue
            if phrase not in seed_entity_dict and phrase in all_phrase_vec_dict:
                distance_arr, entity_index_arr = knn_clf.kneighbors(np.array([all_phrase_vec_dict[phrase]]), 1)
                phrase_entity_dict[phrase] = (entity_name_list[entity_index_arr[0][0]], distance_arr[0][0])

        return phrase_entity_dict


