import os
#定义你要创建的文件夹列表
folders = [
    'data',#放数据
    'src',#放代码
    'results',#放预测图表和结果
    'docs'#放任务书和文档
]
def create_structure():
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"成功创建文件夹{folder}")
        else:
            print(f'文件夹已存在{folder}')

    #顺便创建一个新的main
    if not os.path.exists('main.py'):
        with open('main.py','w',encoding='utf-8') as f:
            f.write("#项目主入口\nprint('项目初始化完成')")

if __name__ == '__main__':
    create_structure()