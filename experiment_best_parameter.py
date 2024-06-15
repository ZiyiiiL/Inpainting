import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tabulate import tabulate
from pathlib import Path
import re

## find the best checkpoint for each experiment and save its metrics to a CSV file
# 路径到TensorBoard日志文件
indir = '/scratch2/ziyliu/LAMA/for_euler/euler_results/'

experiments = ['ziyliu_2024-04-13_02-02-20_train_config_V1.yaml_test', 'ziyliu_2024-04-13_15-18-25_train_config_1e-8_fixed_1e-8_1e-9', 'ziyliu_2024-04-13_15-25-07_train_config_5e-8_fixed_5e-8_5e-9', 'ziyliu_2024-04-13_15-27-38_train_config_1e-7_fixed_1e-7_1e-8', 'ziyliu_2024-04-13_15-30-04_train_config_5e-7_fixed_5e-7_5e-8', 'ziyliu_2024-04-13_15-35-51_train_config_1e-6_fixed_1e-6_1e-7', 'ziyliu_2024-04-13_16-14-51_train_config_5e-6_fixed_5e-6_5e-7', 'ziyliu_2024-04-13_16-19-40_train_config_1e-5_fixed_1e-5_1e-6', 'ziyliu_2024-04-14_00-36-56_train_config_1e-9_fixed_1e-9_1e-10', 'ziyliu_2024-04-14_00-37-32_train_config_5e-9_fixed_5e-9_5e-10', 'ziyliu_2024-04-15_17-59-45_train_config_1e-3_fixed_1e-3_1e-4', 'ziyliu_2024-04-16_17-55-04_train_config_1e-2_fixed_1e-2_1e-3', 'ziyliu_2024-04-16_17-57-12_train_config_9e-3_fixed_9e-3_9e-4', 'ziyliu_2024-04-16_17-58-39_train_config_5e-3_fixed_5e-3_5e-4', 'ziyliu_2024-04-16_17-59-58_train_config_3e-3_fixed_3e-3_3e-4', 'ziyliu_2024-04-16_18-01-57_train_config_9e-4_fixed_9e-4_9e-5', 'ziyliu_2024-04-16_18-04-41_train_config_5e-4_fixed_5e-4_5e-5', 'ziyliu_2024-04-16_22-56-35_train_config_3e-4_fixed_3e-4_3e-5', 'ziyliu_2024-04-17_20-21-14_train_config_1e-4_fixed_1e-4_1e-5', 'ziyliu_2024-04-18_15-06-57_train_config_5e-5_fixed_5e-5_5e-6', 'ziyliu_2024-04-19_13-35-10_train_config_7e-4_fixed_7e-4_7e-5', 'ziyliu_2024-04-19_13-56-07_train_config_7e-3_fixed_7e-3_7e-4', 'ziyliu_2024-04-19_14-05-40_train_config_2e-3_fixed_2e-3_2e-4']

final = pd.DataFrame()

def extract_epoch(filename):
    match = re.search(r'epoch=(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')

for experiment in experiments:
    log_folder = os.path.join(indir, 'tb_logs', experiment)
    log_path = None
    for file in os.listdir(os.path.join(log_folder, 'version_0')):
        if file.startswith('events.out.tfevents'):
            log_path = os.path.join(log_folder, 'version_0', file)
            break

    models = sorted([model.name for model in Path(os.path.join(indir, 'experiments', experiment, 'models')).rglob('*') if model.is_file()], key=extract_epoch)
    model = models[-2]

    # epoch=189-step=32869.ckpt从中提取出步骤编号
    step_number = int(model.split('=')[-1].split('.')[0])
    
    event_acc = EventAccumulator(log_path, size_guidance={'scalars': 0})
    event_acc.Reload()
    scalars = event_acc.Tags()['scalars']
    df = pd.DataFrame()

    for tag in scalars:
        events = event_acc.Scalars(tag)
        tag_data = {'Step': [e.step for e in events],
                    tag: [e.value for e in events]}
        df_tag = pd.DataFrame(tag_data).set_index('Step')
        df = df.join(df_tag, how='outer')  # 以步骤为索引合并DataFrame
    
    df = df.sort_index()

    # 找到'Step'列中数值为step_number的行的第一行
    row = df[df.index == step_number].iloc[0]

    # 添加实验名称和模型名称，从
    row['Experiment'] = experiment.split('_')[-2]

    # 将所有experiments找到的行添加到final DataFrame中
    final = final.append(row)

# 将DataFrame保存为CSV文件
output_file = '/scratch2/ziyliu/LAMA/for_euler/euler_results/output_metrics/final.csv'
final.to_csv(output_file, sep='\t', index=True)  # 使用制表符作为分隔符
print(f"Metrics saved to {output_file}")



###############################################################
# Using Tabulate

# # 将表格保存到文本文件中
# output_file = '/scratch2/ziyliu/LAMA/for_euler/euler_results/output_metrics/V1.txt'

# # 创建一个EventAccumulator实例来加载数据
# event_acc = EventAccumulator(log_path, size_guidance={'scalars': 0})
# event_acc.Reload()  # 加载所有数据

# # 获取所有的标量指标
# scalars = event_acc.Tags()['scalars']

# # 创建一个空的DataFrame
# df = pd.DataFrame()

# # 遍历每个标签并提取数据
# for tag in scalars:
#     events = event_acc.Scalars(tag)
#     tag_data = {'Step': [e.step for e in events],
#                 tag: [e.value for e in events]}
#     df_tag = pd.DataFrame(tag_data).set_index('Step')
#     df = df.join(df_tag, how='outer')  # 以步骤为索引合并DataFrame

# # 排序DataFrame
# # df = df.sort_index()

# # # 使用tabulate格式化数据
# # table = tabulate(df, headers='keys', tablefmt='grid', showindex="always")

# # # 将表格保存到文本文件中
# # with open(output_file, 'w') as f:
# #     f.write(table)

# # print(f"Metrics saved to {output_file}")

# df = df.sort_index()

# # 将DataFrame保存为CSV文件
# # 使用tab分割，保存到TSV文件中
# df.to_csv(output_file, sep='\t', index=True)  # 使用制表符作为分隔符

# print(f"Metrics saved to {output_file}")