# encoding: utf-8


class Trie(object):
    """
    字典树
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        重置字典树
        :return:
        """
        self.root = {}
        self.end = -1

    def insert(self, word):
        """
        插入节点
        :param word:
        :return:
        """
        cur_node = self.root
        for c in word:
            if c not in cur_node:
                cur_node[c] = {}
            cur_node = cur_node[c]

        cur_node[self.end] = True

    def build_trie(self, entity_list, min_len=2):
        """
        构建实体字典树
        :param entity_list: 实体名称列表
        :param min_len: 插入实体的最小长度
        :return:
        """
        for entity in entity_list:
            if len(entity) >= min_len:
                self.insert(entity)

    def search(self, word):
        """
        查找某个词是否存在
        :param word:
        :return:
        """
        cur_node = self.root
        for c in word:
            if c not in cur_node:
                return False
            cur_node = cur_node[c]

        # 至少保证一个节点插入
        if self.end not in cur_node:
            return False

        return True

    def is_prefix(self, prefix):
        """
        查找某个前缀是否存在
        :param prefix:
        :return:
        """
        cur_node = self.root
        for c in prefix:
            if c not in cur_node:
                return False
            cur_node = cur_node[c]

        return True

    def search_entity(self, text):
        """
        最大前缀匹配搜索实体
        :param text:
        :return:
        """
        i = 0
        text_len = len(text)
        entity_list = []
        while i < text_len:
            flag = True
            _tmp = None
            _tmp_i = None
            curr_i = i
            if curr_i > 0 and (text[curr_i - 1].isalpha() or text[curr_i - 1].isdigit()):
                i += 1
                continue

            for j in range(curr_i + 1, text_len + 1):
                word = text[curr_i:j]
                if j < text_len and (text[j].isalpha() or text[j].isdigit()):
                    continue
                if not self.is_prefix(word):
                    break
                if self.search(word):
                    _tmp = word
                    _tmp_i = curr_i
                    flag = False
                    i = j
            if _tmp:
                entity_obj = {}
                entity_obj["form"] = _tmp
                entity_obj["offset"] = _tmp_i
                entity_obj["length"] = len(_tmp)
                entity_list.append(entity_obj)
            if flag:
                i += 1

        return entity_list




if __name__ == "__main__":
    pass

    # word_list = ["国家", "家主", "主席", "平出", "席政", "习近平", "出席"]
    #
    # trie = Trie()
    # trie.build_trie(word_list)
    #
    # text = u"国家主席习近平出席政治局会议"
    # print(text, trie.search_entity(text))
    # print(text, trie.search_entity2(text))



