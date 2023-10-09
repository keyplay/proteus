import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


folder_path = './data/count'
source = 'win7'
target_train_size = 50
random_seed = 55
top_num = 50
threshold = 42795 
num_leaf = 5

def evaluation_binary(true, pred):
  ret_str = []
  ret = []  

  acc = accuracy_score(true, pred)
  s = f'accuracy: {acc:.3%}'
  ret_str.append(s)
  ret.append(acc)
    
  f1 = f1_score(true, pred, average='binary', zero_division=0)
  s = f'f1: {f1:.3%}'
  ret_str.append(s)
  ret.append(f1)
    
  prec = precision_score(true, pred, average='binary', zero_division=0)
  recall = recall_score(true, pred, average='binary', zero_division=0)
  s = f'precision: {prec:.3%}'
  ret_str.append(s)
  s = f'recall: {recall:.3%}'
  ret_str.append(s)
  ret.append(prec)
  ret.append(recall)
  
  
  ret_str.append('\n')
  ret_str = '\t'.join(ret_str)
  
  return ret_str, ret

win7_data = np.load(os.path.join(folder_path, 'win7_data.npy'), allow_pickle=True)
win7_label = np.load(os.path.join(folder_path, 'win7_label.npy'), allow_pickle=True)
win7_class = np.load(os.path.join(folder_path, 'win7_class.npy'), allow_pickle=True)
win7_fileNo = np.load(os.path.join(folder_path, 'win7_fileNo.npy'), allow_pickle=True)
win10_data = np.load(os.path.join(folder_path, 'win10_data.npy'), allow_pickle=True)
win10_label = np.load(os.path.join(folder_path, 'win10_label.npy'), allow_pickle=True)
win10_class = np.load(os.path.join(folder_path, 'win10_class.npy'), allow_pickle=True)
win10_fileNo = np.load(os.path.join(folder_path, 'win10_fileNo.npy'), allow_pickle=True)


win7_combination_raw = np.column_stack((win7_data, win7_class, win7_fileNo))
win10_combination_raw = np.column_stack((win10_data, win10_class, win10_fileNo))

print('win7 raw data shape', win7_data.shape, win7_label.shape)
print('win10 raw data shape', win10_data.shape, win10_label.shape)

win7_class0_idx = win7_label == 0
win7_class1_idx = win7_label == 1


_, win7_class0_new_idx, win10_class0_idx = np.intersect1d(win7_combination_raw[win7_class0_idx, -1][:threshold], win10_combination_raw[:, -1], return_indices=True)
_, win7_class1_new_idx, win10_class1_idx = np.intersect1d(win7_combination_raw[win7_class1_idx, -1], win10_combination_raw[:, -1], return_indices=True)

win7_label = np.concatenate((win7_label[win7_class0_idx][:threshold][win7_class0_new_idx], win7_label[win7_class1_idx][win7_class1_new_idx]))
win7_combination = np.concatenate((win7_combination_raw[win7_class0_idx][win7_class0_new_idx], win7_combination_raw[win7_class1_idx][win7_class1_new_idx]))
win10_label = np.concatenate((win10_label[win10_class0_idx], win10_label[win10_class1_idx]))
win10_combination = np.concatenate((win10_combination_raw[win10_class0_idx], win10_combination_raw[win10_class1_idx]))

print('Source domain:', source)
print('win7 data size', win7_label.shape)
print('win10 data size', win10_label.shape)
print('win7 label and counts', np.unique(win7_label, return_counts=True))
print('win10 label and counts', np.unique(win10_label, return_counts=True))
print('win7 class and counts', np.unique(win7_class, return_counts=True))
print('win10 class and counts', np.unique(win10_class, return_counts=True))


if source == 'win7':
  X_source, y_source = win7_combination, win7_label
  X_target, y_target = win10_combination, win10_label
else:
  X_source, y_source = win10_combination, win10_label
  X_target, y_target = win7_combination, win7_label


X_source_train_combine, X_source_test_combine, y_source_train, y_source_test = train_test_split(X_source, y_source, test_size=0.2, random_state=random_seed)
X_target_train_combine, X_target_test_combine, y_target_train, y_target_test = train_test_split(X_target, y_target, test_size=0.2, random_state=random_seed)
X_target_train_combine, _, y_target_train, _ = train_test_split(X_target_train_combine, y_target_train, train_size=target_train_size, random_state=random_seed)


X_source_train, X_source_test = X_source_train_combine[:, :-2].astype(np.float), X_source_test_combine[:, :-2].astype(np.float)
class_source_train, class_source_test = X_source_train_combine[:, -2], X_source_test_combine[:, -2]
fileNo_source_train, fileNo_source_test = X_source_train_combine[:, -1], X_source_test_combine[:, -1]

X_target_train, X_target_test = X_target_train_combine[:, :-2].astype(np.float), X_target_test_combine[:, :-2].astype(np.float)
class_target_train, class_target_test = X_target_train_combine[:, -2], X_target_test_combine[:, -2]
fileNo_target_train, fileNo_target_test = X_target_train_combine[:, -1], X_target_test_combine[:, -1]

# ---------- find corresponding source samples, given limited target samples -----------
source_pair_idxs = []
target_pair_idxs = []
for i, target_file in enumerate(fileNo_target_train):
  s_idx = np.where(fileNo_source_train == target_file)
  if len(s_idx[0]) > 0: 
    source_pair_idxs.append(s_idx[0][0])
    target_pair_idxs.append(i)
  
print('Size of source training', len(y_source_train), ',Size of source testing', len(y_source_test))
print('Size of target training', len(target_pair_idxs), ',Size of target testing', len(y_target_test))
print('Target label:', y_target_train[target_pair_idxs])
print('Class of target training', class_target_train[target_pair_idxs])


# ------------- adapt source data to target domain ----------------
M, residuals, rank, s = np.linalg.lstsq(X_source_train[source_pair_idxs], X_target_train[target_pair_idxs])
print('residuals:', np.sum(residuals))
transformed_X_source = np.matmul(X_source_train, M)

# ------------- train classifier ----------------
clf = RandomForestClassifier(min_samples_leaf=num_leaf)
clf.fit(transformed_X_source, y_source_train)

clf_tgt_full = RandomForestClassifier(min_samples_leaf=num_leaf)
clf_tgt_full.fit(X_target_train, y_target_train)

clf_source = RandomForestClassifier(min_samples_leaf=num_leaf)
clf_source.fit(X_source_train, y_source_train)
  
# ------ feature importance explanation -----------
start_quantile = 0.7
end_quantile = 0.96
major_fea_tgt_trans_diff_list = []
major_fea_tgt_src_diff_list = []
for threshold in np.arange(start_quantile, end_quantile, 0.01):
    fea_src=clf_source.feature_importances_
    fea_tgt=clf_tgt_full.feature_importances_
    fea_trans=clf.feature_importances_

    fea_trans[fea_trans<np.quantile(fea_trans, threshold)]=0
    fea_src[fea_src<np.quantile(fea_src, threshold)]=0
    fea_tgt[fea_tgt<np.quantile(fea_tgt, threshold)]=0

    major_fea_tgt_trans_diff = np.sum(np.abs(fea_tgt-fea_trans))
    major_fea_tgt_src_diff = np.sum(np.abs(fea_tgt-fea_src))

    major_fea_tgt_trans_diff_list.append(major_fea_tgt_trans_diff)
    major_fea_tgt_src_diff_list.append(major_fea_tgt_src_diff)

plt.figure()   
plt.rcParams.update({'font.size': 12}) 
plt.plot(range(0, len(major_fea_tgt_trans_diff_list)), major_fea_tgt_trans_diff_list, linewidth=3.5, color='royalblue', label='target-adapted_source')
plt.plot(range(0, len(major_fea_tgt_src_diff_list)), major_fea_tgt_src_diff_list, linewidth=3.5, color='crimson', label='target-source')
plt.xticks(range(0, len(major_fea_tgt_src_diff_list), 5), [f"{val:.2f}" for val in np.arange(start_quantile, end_quantile, 0.05)])
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.title(r"Win10$\rightarrow$Win7")
plt.xlabel('Quantile of Feature Importance')
plt.ylabel('Difference of Feature Importance')
plt.show()


# ------------- evaluation -----------
pred = clf.predict(X_target_test)
result_str, result = evaluation_binary(y_target_test, pred)
print('transformation matrix, target to source, training with transferred source data')
print(result_str)



