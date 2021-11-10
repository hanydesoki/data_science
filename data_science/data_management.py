import os
import pandas as pd

class DataGroup:
    '''Read and store all data files from a specific folder with pandas'''
    
    reader = {'csv': pd.read_csv,
            'xls': pd.read_excel,
            'xlsx': pd.read_excel}

    def __init__(self, data_folder, extension='csv'):
        self.data_folder = data_folder
        if extension not in self.reader:
            raise Exception(f'{self.__class__.__name__} does not support {extension} extension.')
        self.extension = extension
        self.files = os.listdir(data_folder)

        self.datas = {}
        self.assembled_data = None

        for data_file in self.files:
            if not data_file.lower().endswith('.' + extension):
                raise Exception(f'{data_file} is not a {extension} file.')
                
            filename = data_file.split('.')[0]
            path = os.path.join(self.data_folder, data_file)
            print(f'Loading file: {data_file}')
            self.datas[filename] = self.reader[extension](path)

        self.check_main_table = False

    def set_main_table(self, fname):
        '''Set a main table in a global table'''
        self.main_table = self[fname]
        self.assembled_data = self.main_table.copy()
        self.check_main_table = True

    def join(self, fname, column):
        '''Join a table with the global table'''
        if not self.check_main_table:
            raise Exception('No main table is set. Use set_main_table method.')

        self.assembled_data = self.assembled_data.merge(self[fname], on=column)

    def get_assembled_data(self):
        return self.assembled_data

    @property
    def filenames(self):
        return list(self.datas)

    def __getitem__(self, item):
        if item not in self.filenames:
            raise IndexError(f'{item} not in {self.__class__.__name__}')

        return self.datas[item]

    def __iter__(self):
        return iter(list(self.datas.items()))

    def __repr__(self):
        return f"{self.__class__.__name__}(data_folder={self.data_folder}, extension='{self.extension}')"

