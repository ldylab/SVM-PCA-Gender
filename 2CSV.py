import pandas as pd
import io
import warnings

warnings.filterwarnings('ignore') # Pandas转化时会有一个Warning，但是不影响，去掉了Warning警告

def FileToCsv(original_file_name, to_csv_file_name):
    with open(original_file_name, 'r') as f:
        lines = f.readlines()
        fo = io.StringIO()

        fo.writelines(u"" + line.replace('(', ',', 5) for line in lines)
        fo.seek(0)

    df = pd.read_csv(fo, sep=',')

    df['Sex'] = df['Sex'].str.replace('_sex', '')
    df['Sex'] = df['Sex'].str.replace(')', '')
    df['Sex'] = df['Sex'].str.replace('^(\s+)|(\s+)$', '')

    df['Age'] = df['Age'].str.replace('_age', '')
    df['Age'] = df['Age'].str.replace(')', '')
    df['Age'] = df['Age'].str.replace('^(\s+)|(\s+)$', '')

    df['Race'] = df['Race'].str.replace('_race', '')
    df['Race'] = df['Race'].str.replace(')', '')
    df['Race'] = df['Race'].str.replace('^(\s+)|(\s+)$', '')

    df['Face'] = df['Face'].str.replace('_face', '')
    df['Face'] = df['Face'].str.replace(')', '')
    df['Face'] = df['Face'].str.replace('^(\s+)|(\s+)$', '')

    df['Prop'] = df['Prop'].str.replace('_prop', '')
    df['Prop'] = df['Prop'].str.replace('[^\u4e00-\u9fa5^,^a-z^A-Z^0-9^\s^\n]', '')
    df['Prop'] = df['Prop'].str.replace('^(\s+)|(\s+)$', '')

    df.to_csv(to_csv_file_name + '.csv', sep=',', index=None)
    print(to_csv_file_name + '.csv' + ' is OK')

FileToCsv('faceDR', 'CookFaceDR')
FileToCsv('faceDS', 'CookFaceDS')
