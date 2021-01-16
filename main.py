

from util.file_util import FileUtil


if __name__ == "__main__":


    data_path = "/data/fangzheng/bert_autoner/laptop/source_data/laptop_test.json"

    text_obj_list = FileUtil.read_text_obj_data(data_path)

    for text_obj in text_obj_list:
        # text_obj.pop("text_pos")
        for entity_obj in text_obj["entity_list"]:
            entity_obj["offset"] = int(entity_obj["offset"])

    FileUtil.save_text_obj_data(text_obj_list, data_path)





