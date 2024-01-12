import json
import argparse

class Config:
    def __init__(self, config_file=None, args=None):
        self.config = {}  # 初始化空字典

        if config_file:
            with open(config_file, 'r', encoding='utf-8') as file:
                config = json.load(file)
                self.config.update(config)
        
        if args:
            self.update_config_with_args(args)
        
    # 自定义方法, 从列表中获取参数
    def update_config_with_args(self, args):
        for key, value in vars(args).items():
            if value is not None:
                self.config[key] = value
    
    def replace_project_name_placeholder(self):
        run_name = self.config.get('run_name')
        if run_name:
            for key, value in self.config.items():
                if isinstance(value, str) and '{run_name}' in value:
                    # .format() 方法通常使用大括号 {} 来表示要替换的占位符，不需要{}包围变量名
                    self.config[key] = value.format(run_name=run_name)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __str__(self):
        return str(self.config)
    
    def get_config():
        parser = argparse.ArgumentParser()

        parser.add_argument('--config', default='/home/swh/text_classification/utils/config.json', type=str, help='Path to config file')
        parser.add_argument('--embed_dim', type=int, help='Embedding dimension')
        parser.add_argument('--num_filters', type=int, help='Number of filters')
        parser.add_argument('--filter_sizes', type=int, nargs='+', help='Filter sizes')
        parser.add_argument('--num_classes', type=int, help='Number of classes')
        parser.add_argument('--max_length', type=int, help='Max length')
        parser.add_argument('--batch_size', type=int, help='Batch size')
        parser.add_argument('--num_epochs', type=int, help='Number of epochs')
        parser.add_argument('--learning_rate', type=float, help='Learning rate')
        parser.add_argument('--project_name', type=str, help='Project name')
        parser.add_argument('--run_name', type=str, help='Run name')
        parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
        parser.add_argument('--vocab_path', type=str, help='Path to the vocabulary')
        parser.add_argument('--model_path', type=str, help='Path to the model')
        parser.add_argument('--result_save_path', type=str, help='Path to save the results')

        args = parser.parse_args()
        return Config(config_file=args.config, args=args)
