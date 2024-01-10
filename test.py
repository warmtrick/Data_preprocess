from utils.config import Config


# config = Config.get_config()

# print(config)
# print(config.get('embed_dim'))

if __name__ == "__main__":
    aconfig = Config.get_config()  # 使用 get_config 静态方法初始化 Config

    print(aconfig.config)  # 打印配置信息

    embed_dim = aconfig.get('embed_dim')
    num_filters = aconfig.get('num_filters')
    batch_size = aconfig.get('batch_size')

    print(embed_dim, num_filters, batch_size)
