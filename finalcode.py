# coding: utf-8
import pandas as pd
import numpy as np
import math
import os
import random
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

def partition(ls, size):
    return [ls[i:i + size] for i in range(0, len(ls), size)]


def NegativeGenerate(DrugDisease, AllDurg, AllDisease):
    NegativeSample = []
    counterN = 0
    while counterN < len(DrugDisease):
        counterR = random.randint(0, len(AllDurg) - 1)
        counterD = random.randint(0, len(AllDisease) - 1)
        DiseaseAndRnaPair = []
        DiseaseAndRnaPair.append(AllDurg[counterR])
        DiseaseAndRnaPair.append(AllDisease[counterD])
        flag1 = 0
        counter = 0
        while counter < len(DrugDisease):
            if DiseaseAndRnaPair == DrugDisease[counter]:
                flag1 = 1
                break
            counter = counter + 1
        if flag1 == 1:
            continue
        flag2 = 0
        counter1 = 0
        while counter1 < len(NegativeSample):
            if DiseaseAndRnaPair == NegativeSample[counter1]:
                flag2 = 1
                break
            counter1 = counter1 + 1
        if flag2 == 1:
            continue
        if (flag1 == 0 & flag2 == 0):
            NamePair = []
            NamePair.append(AllDurg[counterR])
            NamePair.append(AllDisease[counterD])
            NegativeSample.append(NamePair)
            counterN = counterN + 1
    return NegativeSample


def main(options):


    # DrDiNum = pd.read_csv('data2/dataset/3、di-dr.csv', header=None)
    DrDiNum = pd.read_csv('finaldata/data/4.药物-疾病id关联.csv', header=None)
    DrGeNum = pd.read_csv('finaldata/data/6.药物-基因id关联.csv', header=None)
    DiGeNum = pd.read_csv('finaldata/data/5.基因-疾病id关联.csv', header=None)

    RandomList = random.sample(range(0, len(DrDiNum)), len(DrDiNum))
    print('len(RandomList)', len(RandomList))
    NewRandomList = partition(RandomList, math.ceil(len(RandomList) / options.fold_num))
    print('len(NewRandomList[0])', len(NewRandomList[0]))
    NewRandomList = pd.DataFrame(NewRandomList)
    NewRandomList = NewRandomList.fillna(int(0))
    NewRandomList = NewRandomList.astype(int)
    NewRandomList.to_csv('finaldata/processing-data/NewRandomList.csv', header=None, index=False)
    del NewRandomList, RandomList

    Nindex = pd.read_csv('finaldata/processing-data/NewRandomList.csv', header=None)
    for i in range(len(Nindex)):
        kk = []
        for j in range(options.fold_num):
            if j != i:
                kk.append(j)
        index = np.hstack(
            [np.array(Nindex)[kk[0]], np.array(Nindex)[kk[1]], np.array(Nindex)[kk[2]], np.array(Nindex)[kk[3]],
             np.array(Nindex)[kk[4]],
             np.array(Nindex)[kk[5]], np.array(Nindex)[kk[6]], np.array(Nindex)[kk[7]], np.array(Nindex)[kk[8]]])
        DTIs_train = pd.DataFrame(np.array(DrDiNum)[index])
        DTIs_train.to_csv('finaldata/processing-data/DrDiIs_train' + str(i) + '.csv', header=None, index=False)
        DTIs_test = pd.DataFrame(np.array(DrDiNum)[np.array(Nindex)[i]])
        DTIs_test.to_csv('finaldata/processing-data/DrDiIs_test' + str(i) + '.csv', header=None, index=False)
        print(i)
    del Nindex, index, DTIs_train, DTIs_test

    DTIs_train = DrDiNum.append(DrGeNum.append(DiGeNum))
    DTIs_train = DTIs_train.sample(frac=1.0)
    DTIs_train.to_csv('finaldata/processing-data/AllDrDiIs_train.txt', sep='\t', header=None, index=False)
    Di = pd.read_csv('finaldata/data/1.疾病id-name.csv', header=None, names=['id', 'name'])
    Dr = pd.read_csv('finaldata/data/2.药物id-name.csv', header=None, names=['id', 'name'])
    Pr = pd.read_csv('finaldata/data/3.基因id-name.csv', header=None, names=['id', 'name'])
    NegativeSample = NegativeGenerate(DrDiNum.values.tolist(), Dr['id'].values.tolist(), Di['id'].values.tolist())
    NegativeSample = pd.DataFrame(NegativeSample)
    NegativeSample.to_csv('finaldata/processing-data/NegativeSample.csv', header=None, index=False)

    # creat_var = globals()
    creat_var = globals()
    Negative = pd.read_csv('finaldata/processing-data/NegativeSample.csv', header=None)
    Nindex = pd.read_csv('finaldata/processing-data/NewRandomList.csv', header=None)
    Attribute = pd.read_csv('finaldata/data/all-node-attribute(withid).csv', header=None, index_col=0)
    Attribute = Attribute.iloc[:, 1:]
    Embedding = pd.read_csv('finaldata/data/node2vec_embeddings.txt', sep=' ', header=None, skiprows=1)
    Embedding = Embedding.sort_values(0, ascending=True)
    Embedding.set_index([0], inplace=True)
    Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
    for i in range(options.fold_num):
        train_data = pd.read_csv('finaldata/processing-data/DrDiIs_train' + str(i) + '.csv', header=None)
        train_data[2] = train_data.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
        kk = []
        for j in range(10):
            if j != i:
                kk.append(j)
        index = np.hstack(
            [np.array(Nindex)[kk[0]], np.array(Nindex)[kk[1]], np.array(Nindex)[kk[2]], np.array(Nindex)[kk[3]],
             np.array(Nindex)[kk[4]],
             np.array(Nindex)[kk[5]], np.array(Nindex)[kk[6]], np.array(Nindex)[kk[7]], np.array(Nindex)[kk[8]]])
        result = train_data.append(pd.DataFrame(np.array(Negative)[index]))
        labels_train = result[2]
        data_train_feature = pd.concat([pd.concat(
            [Attribute.loc[result[0].values.tolist()], Embedding.loc[result[0].values.tolist()]], axis=1).reset_index(
            drop=True),
                                        pd.concat([Attribute.loc[result[1].values.tolist()],
                                                   Embedding.loc[result[1].values.tolist()]], axis=1).reset_index(
                                            drop=True)], axis=1)

        '''data_train_feature = pd.concat(
            [Embedding.loc[result[0].values.tolist()].reset_index(drop=True),
             Embedding.loc[result[1].values.tolist()].reset_index(drop=True)], axis=1)'''
        creat_var['data_train' + str(i)] = data_train_feature.values.tolist()
        creat_var['labels_train' + str(i)] = labels_train
        print(len(labels_train))
        del labels_train, result, data_train_feature

        test_data = pd.read_csv('finaldata/processing-data/DrDiIs_test' + str(i) + '.csv', header=None)
        test_data[2] = test_data.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
        result = test_data.append(pd.DataFrame(np.array(Negative)[np.array(Nindex)[i]]))
        labels_test = result[2]
        data_test_feature = pd.concat([pd.concat(
            [Attribute.loc[result[0].values.tolist()], Embedding.loc[result[0].values.tolist()]], axis=1).reset_index(
            drop=True),
            pd.concat([Attribute.loc[result[1].values.tolist()],
                       Embedding.loc[result[1].values.tolist()]], axis=1).reset_index(
                drop=True)], axis=1)
        creat_var['data_test' + str(i)] = data_test_feature.values.tolist()
        creat_var['labels_test' + str(i)] = labels_test
        print(len(labels_test))
        del train_data, test_data, labels_test, result, data_test_feature
        print(i)

    data_train = [data_train0, data_train1, data_train2, data_train3, data_train4, data_train5, data_train6,
                  data_train7, data_train8, data_train9]
    data_test = [data_test0, data_test1, data_test2, data_test3, data_test4, data_test5, data_test6, data_test7,
                 data_test8, data_test9]
    labels_train = [labels_train0, labels_train1, labels_train2, labels_train3, labels_train4, labels_train5,
                    labels_train6, labels_train7, labels_train8, labels_train9]
    labels_test = [labels_test0, labels_test1, labels_test2, labels_test3, labels_test4, labels_test5, labels_test6,
                   labels_test7, labels_test8, labels_test9]

    print(str(options.fold_num) + "-CV")

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    AllResult = []
    fig_roc = plt.figure()
    fig_pr = plt.figure()
    ax_roc = fig_roc.add_subplot(111)
    ax_pr = fig_pr.add_subplot(111)
    correlation = {}

    precision_scores = []
    recall_scores = []
    f1_scores = []
    mcc_scores = []

    for i in range(10):
        X_train, X_test = data_train[i], data_test[i]
        Y_train, Y_test = np.array(labels_train[i]), np.array(labels_test[i])
        best_RandomF = RandomForestClassifier(n_estimators=options.tree_number)
        best_RandomF.fit(np.array(X_train), np.array(Y_train))
        y_score0 = best_RandomF.predict(np.array(X_test))
        y_score_RandomF = best_RandomF.predict_proba(np.array(X_test))
        fpr, tpr, thresholds = roc_curve(Y_test, y_score_RandomF[:, 1])
        precision, recall, _ = precision_recall_curve(Y_test, y_score_RandomF[:, 1])
        ax_roc.plot(fpr, tpr)
        ax_pr.plot(recall, precision)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # auc
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        print('ROC fold %d(AUC=%0.4f)' % (i, roc_auc))

        y_pred = np.argmax(y_score_RandomF, axis=1)
        precision = precision_score(Y_test, y_pred)
        recall = recall_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        mcc = matthews_corrcoef(Y_test, y_pred)
        # 输出每个折叠的结果
        print('Fold %d:' % i)
        print('Precision: %0.4f' % precision)
        print('Recall: %0.4f' % recall)
        print('F1 score: %0.4f' % f1)
        print('Matthews Correlation Coefficient: %0.4f' % mcc)
        print()
        # 保存结果用于计算十次折叠的均值
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        mcc_scores.append(mcc)

        test = pd.read_csv('finaldata/processing-data/DrDiIs_test' + str(i) + '.csv', header=None, index_col=None)
        test = np.array(test)
        for cnt, line in enumerate(y_score_RandomF[:47]):
            correlation[str(test[int(cnt)][0])] = correlation.get(str(test[int(cnt)][0]), {})
            correlation[str(test[int(cnt)][0])][str(test[int(cnt)][1])] = correlation[str(test[int(cnt)][0])].get(
                str(test[int(cnt)][1]), line[1])
    correlation_df = pd.DataFrame.from_dict(correlation, orient='index')

    Drname = dict(zip(Dr['id'], Dr['name']))
    Diname = dict(zip(Di['id'], Di['name']))
    print(Drname)
    print(Diname)
    for drug in correlation:
        dicta = correlation[drug]
        dicta = dict(sorted(dicta.items(), key=lambda item: item[1], reverse=True))
        list = []
        for i in dicta:
            list.append(Diname[int(i)])
        print('与药物{}关联的疾病相关度排序为：{}'. format(Drname[int(drug)], str(list)))
    correlation_df.to_csv('correlation_.csv')
    # fig_roc.savefig('_ROC.png')
    # fig_pr.savefig('P-R.png')

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('Mean ROC (AUC=%0.4f)' % (mean_auc))

    # np.savetxt(r'D:\文献文献\相关文献\论文文文\paper\图和表\ROC-DATA\RF实验mean_fpr.txt', mean_fpr)
    # np.savetxt(r'D:\文献文献\相关文献\论文文文\paper\图和表\ROC-DATA\RF实验mean_tpr.txt', mean_tpr)
    # # Save mean_auc to text file
    # np.savetxt(r'D:\文献文献\相关文献\论文文文\paper\图和表\ROC-DATA\RF实验mean_auc.txt', [mean_auc])

    precision_mean = np.mean(precision_scores)
    recall_mean = np.mean(recall_scores)
    f1_mean = np.mean(f1_scores)
    mcc_mean = np.mean(mcc_scores)

    print('Mean of 10-fold cross-validation:')
    print('Precision: %0.4f' % precision_mean)
    print('Recall: %0.4f' % recall_mean)
    print('F1 score: %0.4f' % f1_mean)
    print('Matthews Correlation Coefficient: %0.4f' % mcc_mean)
    # pr_data = {
    #     'mean_fpr': mean_fpr,
    #     'precision': precision,
    #     'recall': recall,
    #     'thresholds': thresholds
    # }
    #
    # # 指定保存文件的文件夹路径
    # folder_path = r'D:\文献文献\相关文献\论文文文\paper\图和表'
    # # 确保文件夹存在，如果不存在则创建
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # # 拼接文件完整路径
    # file_path = os.path.join(folder_path, '实验pr_data.txt')
    # # 打开文件进行写入
    # with open(file_path, 'w', encoding='utf-8') as file:
    #     file.write('PR 数据\n')
    #     file.write('平均假阳率 (mean_fpr):\n')
    #     file.write(str(pr_data['mean_fpr']) + '\n\n')
    #     file.write('精确率 (precision):\n')
    #     file.write(str(pr_data['precision']) + '\n\n')
    #     file.write('召回率 (recall):\n')
    #     file.write(str(pr_data['recall']) + '\n\n')
    #     file.write('阈值 (thresholds):\n')
    #     file.write(str(pr_data['thresholds']) + '\n\n')
    # print(f'数据已保存到 {file_path}')

if __name__ == '__main__':
    import optparse
    import sys

    parser = optparse.OptionParser(usage=main.__doc__)
    parser.add_option("-d", "--dataset", action='store',
                      dest='dataset', default=1, type='int',
                      help=('The dataset of cross-validation '
                            '(1: B-Dataset; 2: F-Dataset)'))
    parser.add_option('-f', '--fold num', action='store',
                      dest='fold_num', default=10, type='int',
                      help=('The fold number of cross-validation '
                            '(default: 10)'))

    parser.add_option('-n', '--tree number', action='store',
                      dest='tree_number', default=999, type='int',
                      help=('The number of tree of RandomForestClassifier '
                            '(default: 999)'))

    options, args = parser.parse_args()
    print(options)
    sys.exit(main(options))
