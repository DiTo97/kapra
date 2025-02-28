import numpy as np
from loguru import logger
from pathlib import Path


class AnonymizedDataset:
    def __init__(self, anonymized_data: list = list(), pattern_anonymized_data: dict = dict(),
                 suppressed_data: list = list(), sensitive: dict = {}):
        self.anonymized_data = anonymized_data 
        self.pattern_anonymized_data = pattern_anonymized_data 
        self.suppressed_data = suppressed_data
        self.final_data_anonymized = dict()
        self.sensitive = sensitive


    def construct(self):
        """
        Create dataset ready to be anonymized
        :return:
        """
        logger.info("Start creation dataset anonymized")
        logger.info("Added {} anonymized group".format(len(self.anonymized_data)))
        for index in range(0, len(self.anonymized_data)): 

            k_group = self.anonymized_data[index]
                        
            max_value = np.amax(np.array(list(k_group.values())), 0)
            min_value = np.amin(np.array(list(k_group.values())), 0)

            for key in k_group.keys():
                # key = row product
                self.final_data_anonymized[key] = list()
                value_row = list()
                for column_index in range(0, len(max_value)):
                    value_row.append("[{}|{}]".format(min_value[column_index], max_value[column_index]))
                
                value_row.append(self.pattern_anonymized_data[key]) 
                value_row.append(str(self.sensitive[key]))
                value_row.append("Group: {}".format(index))

                self.final_data_anonymized[key] = value_row
        
        logger.info("Added {} suppressed group".format(len(self.suppressed_data)))
        for index in range(0, len(self.suppressed_data)):
            group = self.suppressed_data[index]
            for key in group.keys():
                value_row = [" - "]*len(group[key])
                value_row.append(" - ") # pattern rapresentation
                value_row.append(" - ") # group
                self.final_data_anonymized[key] = value_row

    def save(self, output_path, col_names):
        logger.info("Saving on file dataset anonymized")
        with open(output_path, "w") as file_to_write:
            file_to_write.write(",".join(col_names) + ',sax,as,group' + "\n")
            value_to_print_on_file = ""
            for key, value in self.final_data_anonymized.items():
                value_to_print_on_file = "{},{}".format(key, ",".join(value))
                file_to_write.write(value_to_print_on_file+"\n")