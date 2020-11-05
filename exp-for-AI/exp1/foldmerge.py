import os
import shutil

# filePath = r'C:\Users\kdjlyy\Desktop\courses\人工智能2020\experiment\data\train'  # 训练集数据文件夹
filePath = r'C:\Users\kdjlyy\Desktop\courses\人工智能2020\experiment\data\valid'  # 验证集数据文件夹
datasetPath = r'C:\Users\kdjlyy\Desktop\courses\人工智能2020\experiment\data\dataset'  # 合并后的文件夹（需提前创建）

# 6149*2 = 12298
# 12298 + 1020*2 = 12298 + 2040 = 14338
# enjoy!

# filePath = r'C:\Users\kdjlyy\Desktop\codes\PythonCode\foldtest'  # 数据文件夹
# datasetPath = r'C:\Users\kdjlyy\Desktop\codes\PythonCode\dataset'  # 合并后的文件夹（需提前创建）

def CreateDir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print('\n', path + '目录创建成功')
    else:
        print('\n', path + '目录已存在')


def readname(path):
    name = os.listdir(path)
    return name

# 创建文件   file_path：文件路径    msg：即要写入的内容
def create_file(file_path, msg):
    f = open(file_path, "a")
    f.write(msg)
    f.close

# def CopyFile(filepath, newPath):
#     # 获取当前路径下的文件名，返回List
#     fileNames = os.listdir(filepath)
#     for file in fileNames:
#         # 将文件命加入到当前文件路径后面
#         newDir = filepath + '\\' + file
#         # 如果是文件
#         if os.path.isfile(newDir):
#             print(newDir)
#             newFile = newPath + file
#             shutil.copyfile(newDir, newFile)
#         #如果不是文件，递归这个文件夹的路径
#         else:
#             CopyFile(newDir,newPath)


if __name__ == "__main__":
    name = readname(filePath)
    # print(name)
    for i in name:
        label = i
        # print("###", label)
        filePath_son = filePath + "\\" + label
        print(filePath_son)
        name_son = readname(filePath_son)
        for j in name_son:
            label_txt = filePath_son + '\\' + j[:-4] + ".txt";
            if os.path.exists(label_txt) == False:
                create_file(label_txt, label)
            # print(label_txt)

        name_son = readname(filePath_son)
        for k in name_son:
            file_name = filePath_son + '\\' + k
            new_file_name = datasetPath + '\\' + k
            if os.path.exists(new_file_name) == False:
                shutil.copy(file_name, new_file_name)
            print(new_file_name)
