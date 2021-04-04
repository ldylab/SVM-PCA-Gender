import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# 输入核函数名称和参数gamma值，返回SVM训练十折交叉验证的准确率
def SVM(kernel_name, param):
    # 十折交叉验证计算出平均准确率
    # n_splits交叉验证，随机取
    kf = KFold(n_splits=10, shuffle=True)
    precision_average = 0.0
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5]}  # 自动穷举出最优的C参数
    clf = GridSearchCV(SVC(kernel=kernel_name, class_weight='balanced', gamma=param),
                       param_grid)
    for train, test in kf.split(X):
        clf = clf.fit(X[train], y[train])
        # print(clf.best_estimator_)
        test_pred = clf.predict(X[test])
        # print classification_report(y[test], test_pred)
        # 计算平均准确率
        precision = 0
        for i in range(0, len(y[test])):
            if (y[test][i] == test_pred[i]):
                precision = precision + 1
        precision_average = precision_average + float(precision) / len(y[test])
    precision_average = precision_average / 10

    return precision_average

all_data_set = []  # 由每一张图片的list所构成
all_data_label = []  # 数据标签（男：标记为1，女：标记为0）

sex_file = u"./ClassifyFiles/CookFaceAllWithoutMissingWithSame.csv"
sex_df = pd.read_csv(sex_file, sep=',')

from_path = u"./rawdata"

for elem in range(len(sex_df)):
    from_file_name = from_path + '/' + str(sex_df["PicNum"][elem])
    img_raw = np.fromfile(from_file_name, dtype=np.uint8) # 读取图片

    all_data_set.append(list(img_raw.tolist()))

    gender = sex_df["Sex"][elem]
    if gender == 'male':
        all_data_label.append(1)
    elif gender == 'female':
        all_data_label.append(0)

# PCA降维
n_components = 80 # 这一个是找到的较优参数（从图像中可知）
pca = PCA(n_components=n_components, svd_solver='auto',
          whiten=True).fit(all_data_set)
# PCA降维后的总数据集
all_data_pca = pca.transform(all_data_set)
eigenfaces = pca.components_.reshape((n_components, 128, 128))
# X为降维后的数据，y是对应类标签
X = np.array(all_data_pca)
y = np.array(all_data_label)

# 老师提供的数据集有很强的规律性，给它打乱下
np.random.seed(120)
np.random.shuffle(X)
np.random.seed(120)
np.random.shuffle(y)

'''
# 输出Eigenfaces
plt.figure("Eigenfaces")
for i in range(1, 81):
    plt.subplot(8, 10, i).imshow(eigenfaces[i-1], cmap="gray")
    plt.xticks(())
    plt.yticks(())
plt.show()
'''

# SVM分类
t0 = time()
param_grid = {'C': [100], 'gamma': [0.01], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
print("\nn_components: " + str(n_components))

scores = cross_val_score(clf, X, y, cv=10)
print("Average_Score = # ", scores.mean())

'''
# 输出核函数与gamma测试图，选择较优参数使用，跑起来会比较久
t0 = time()
n_components = 80
pca = PCA(n_components=n_components, svd_solver='auto',
              whiten=True).fit(all_data_set)
# PCA降维后的总数据集
all_data_pca = pca.transform(all_data_set)
# X为降维后的数据，y是对应类标签
X = np.array(all_data_pca)
y = np.array(all_data_label)
kernel_to_test = ['rbf', 'poly', 'sigmoid']
# rint SVM(kernel_to_test[0], 0.1)
plt.figure(1)

for kernel_name in kernel_to_test:
    x_label = np.linspace(0.0001, 1, 100)
    y_label = []
    for i in x_label:
        y_label.append(SVM(kernel_name, i))
    plt.plot(x_label, y_label, label=kernel_name)

print("done in %0.3fs" % (time() - t0))
plt.xlabel("Gamma")
plt.ylabel("Precision")
plt.title('Different Kernels Contrust')
plt.legend()
plt.show()
'''