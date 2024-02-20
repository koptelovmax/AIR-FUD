import numpy as np

from sklearn.metrics import cohen_kappa_score

f_name_1 = 'user_1'
f_name_2 = 'user_2'
f_name_3 = 'user_3'
f_name_4 = 'user_4'
f_name_5 = 'user_5'
f_name_6 = 'user_6'
f_name_7 = 'user_7'
f_name_8 = 'user_8'
f_name_9 = 'user_9'
#%%
# LOAD DATA:
#%%
f = open("Summaries/survey/"+f_name_1+".csv", "r")

f.readline() # skip headers

u1 = []
for line in f:
    try:
        u1.append([int(i) for i in line.strip().split(',')])
    except ValueError:
        print('Invalid input:',line)
        
f.close()
#%
f = open("Summaries/survey/"+f_name_2+".csv", "r")

f.readline() # skip headers

u2 = []
for line in f:
    try:
        u2.append([int(i) for i in line.strip().split(',')])
    except ValueError:
        print('Invalid input:',line)
        
f.close()
#%
f = open("Summaries/survey/"+f_name_3+".csv", "r")

f.readline() # skip headers

u3 = []
for line in f:
    try:
        u3.append([int(i) for i in line.strip().split(',')])
    except ValueError:
        print('Invalid input:',line)
        
f.close()
#%
f = open("Summaries/survey/"+f_name_4+".csv", "r")

f.readline() # skip headers

u4 = []
for line in f:
    try:
        u4.append([int(i) for i in line.strip().split(',')])
    except ValueError:
        print('Invalid input:',line)
        
f.close()
#%
f = open("Summaries/survey/"+f_name_5+".csv", "r")

f.readline() # skip headers

u5 = []
for line in f:
    try:
        u5.append([int(i) for i in line.strip().split(',')])
    except ValueError:
        print('Invalid input:',line)
        
f.close()
#%
f = open("Summaries/survey/"+f_name_6+".csv", "r")

f.readline() # skip headers

u6 = []
for line in f:
    try:
        u6.append([int(i) for i in line.strip().split(',')])
    except ValueError:
        print('Invalid input:',line)
        
f.close()
#%
f = open("Summaries/survey/"+f_name_7+".csv", "r")

f.readline() # skip headers

u7 = []
for line in f:
    try:
        u7.append([int(i) for i in line.strip().split(',')])
    except ValueError:
        print('Invalid input:',line)
        
f.close()
#%
f = open("Summaries/survey/"+f_name_8+".csv", "r")

f.readline() # skip headers

u8 = []
for line in f:
    try:
        u8.append([int(i) for i in line.strip().split(',')])
    except ValueError:
        print('Invalid input:',line)
        
f.close()
#%
f = open("Summaries/survey/"+f_name_9+".csv", "r")

f.readline() # skip headers

u9 = []
for line in f:
    try:
        u9.append([int(i) for i in line.strip().split(',')])
    except ValueError:
        print('Invalid input:',line)
        
f.close()
#%%
# Definition of collective users:
user_1 = u1 + u2 + u4[:15] + u5[:15] + u6[15:] + u8
user_2 = u7 + u3 + u6[:15] + u4[15:] + u5[15:] + u9
#%
## EXTRACT VERIFIABLE SEGMENTS ONLY:
#user_1 = user_1[0:64]
#user_2 = user_2[0:64]
#%
## EXTRACT SOFT SEGMENTS ONLY:
#user_1 = user_1[64:96]
#user_2 = user_2[64:96]
#%
## EXTRACT NON-VERIFIABLE SEGMENTS ONLY:
#user_1 = user_1[96:]
#user_2 = user_2[96:]
#%%
# Extract attributes w.r.t. each collective user:
user_1_1 = [i[0] for i in user_1]
user_1_2 = [i[1] for i in user_1]
user_1_3 = [i[2] for i in user_1]
user_1_4 = [i[3] for i in user_1]

user_2_1 = [i[0] for i in user_2]
user_2_2 = [i[1] for i in user_2]
user_2_3 = [i[2] for i in user_2]
user_2_4 = [i[3] for i in user_2]
#%%
# EVALUATION:
#
# I. Calculation of Kappa:
#%%
#calculate Cohen's Kappa
kappa_1 = cohen_kappa_score(user_1_1, user_2_1)
kappa_2 = cohen_kappa_score(user_1_2, user_2_2)
kappa_3 = cohen_kappa_score(user_1_3, user_2_3)
kappa_4 = cohen_kappa_score(user_1_4, user_2_4)

print("Kappa: ",np.round(kappa_1,2),np.round(kappa_2,2),np.round(kappa_3,2),np.round(kappa_4,2))
#%%
# Global Kappa:
kappa_alt = cohen_kappa_score(user_1_1+user_1_2+user_1_3+user_1_4, user_2_1+user_2_2+user_2_3+user_2_4)

print("Kappa (global): ",np.round(kappa_alt,2)) # 0.17 correspond to slight agreement
#%%
# Cohen's Kappa (binary):
user_1_1b = [bool(i) for i in user_1_1]
user_1_2b = [bool(i) for i in user_1_2]
user_1_3b = [bool(i) for i in user_1_3]
user_1_4b = [bool(i) for i in user_1_4]
user_2_1b = [bool(i) for i in user_2_1]
user_2_2b = [bool(i) for i in user_2_2]
user_2_3b = [bool(i) for i in user_2_3]
user_2_4b = [bool(i) for i in user_2_4]

kappa_1b = cohen_kappa_score(user_1_1b, user_2_1b)
kappa_2b = cohen_kappa_score(user_1_2b, user_2_2b)
kappa_3b = cohen_kappa_score(user_1_3b, user_2_3b)
kappa_4b = cohen_kappa_score(user_1_4b, user_2_4b)

print("Kappa (binary): ",np.round(kappa_1b,2),np.round(kappa_2b,2),np.round(kappa_3b,2),np.round(kappa_4b,2)) # 0.39 correspond to fair agreement
#%%
# Global Kappa (binary):
kappa_altb = cohen_kappa_score(user_1_1b+user_1_2b+user_1_3b+user_1_4b, user_2_1b+user_2_2b+user_2_3b+user_2_4b)

print("Kappa (global binary): ",np.round(kappa_altb,2)) # 0.39 correspond to fair agreement
#%%
# II. Compute Mean reciprocal rank (MRR) for AMR
#%%
def mean_reciprocal_rank(rs): # taken from: https://www.programcreek.com/python/?CodeExample=mean%20reciprocal%20rank
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs]) 

# Reciprocal ranks:
# 0 --> [0,0,0,0]
# 1 --> [1,0,0,0]
# 2 --> [0,1,0,0]
# 3 --> [0,0,1,0]
# 4 --> [0,0,0,1]

# RR for user 1:
rs_u1 = np.zeros((len(user_1),4), int)
for i in range(len(user_1)):
    if user_1_1[i] != 0:
        rs_u1[i][user_1_1[i]-1] = 1

# RR for user 2:
rs_u2 = np.zeros((len(user_2),4), int)
for i in range(len(user_2)):
    if user_2_1[i] != 0:
        rs_u2[i][user_2_1[i]-1] = 1

# Compute average b/w different users:
mrs_u1 = mean_reciprocal_rank(rs_u1)
print("MRR (user 1):",np.round(mrs_u1,4))

mrs_u2 = mean_reciprocal_rank(rs_u2)
print("MRR (user 2):",np.round(mrs_u2,4))

# Compute average:
print("MRR (mean):",np.round(np.mean([mrs_u1,mrs_u2]),4))
#%%
# MRR for Chat GPT:

# GPT-1:    
# RR for user 1:
rs_g1_u1 = np.zeros((len(user_1),4), int)
for i in range(len(user_1)):
    if user_1_2[i] != 0:
        rs_g1_u1[i][user_1_2[i]-1] = 1

# RR for user 2:
rs_g1_u2 = np.zeros((len(user_2),4), int)
for i in range(len(user_2)):
    if user_2_2[i] != 0:
        rs_g1_u2[i][user_2_2[i]-1] = 1

# Compute average b/w different users:
mrs_g1_u1 = mean_reciprocal_rank(rs_g1_u1)
print("MRR user 1 (GPT-1):",np.round(mrs_g1_u1,4))

mrs_g1_u2 = mean_reciprocal_rank(rs_g1_u2)
print("MRR user 2 (GPT-1):",np.round(mrs_g1_u2,4))

# Compute average:
print("MRR mean (GPT-1):",np.round(np.mean([mrs_g1_u1,mrs_g1_u2]),4),"\n")

# GPT-2:    
# RR for user 1:
rs_g2_u1 = np.zeros((len(user_1),4), int)
for i in range(len(user_1)):
    if user_1_3[i] != 0:
        rs_g2_u1[i][user_1_3[i]-1] = 1

# RR for user 2:
rs_g2_u2 = np.zeros((len(user_2),4), int)
for i in range(len(user_2)):
    if user_2_3[i] != 0:
        rs_g2_u2[i][user_2_3[i]-1] = 1

# Compute average b/w different users:
mrs_g2_u1 = mean_reciprocal_rank(rs_g2_u1)
print("MRR user 1 (GPT-2):",np.round(mrs_g2_u1,4))

mrs_g2_u2 = mean_reciprocal_rank(rs_g2_u2)
print("MRR user 2 (GPT-2):",np.round(mrs_g2_u2,4))

# Compute average:
print("MRR mean (GPT-2):",np.round(np.mean([mrs_g2_u1,mrs_g2_u2]),4),"\n")

# GPT-3:    
# RR for user 1:
rs_g3_u1 = np.zeros((len(user_1),4), int)
for i in range(len(user_1)):
    if user_1_4[i] != 0:
        rs_g3_u1[i][user_1_4[i]-1] = 1

# RR for user 2:
rs_g3_u2 = np.zeros((len(user_2),4), int)
for i in range(len(user_2)):
    if user_2_4[i] != 0:
        rs_g3_u2[i][user_2_4[i]-1] = 1

# Compute average b/w different users:
mrs_g3_u1 = mean_reciprocal_rank(rs_g3_u1)
print("MRR user 1 (GPT-3):",np.round(mrs_g3_u1,4))

mrs_g3_u2 = mean_reciprocal_rank(rs_g3_u2)
print("MRR user 2 (GPT-3):",np.round(mrs_g3_u2,4))

# Compute average:
print("MRR mean (GPT-3):",np.round(np.mean([mrs_g3_u1,mrs_g3_u2]),4),"\n")

print("MRR mean (ChatGPT):",np.round(np.mean([np.mean([mrs_g1_u1,mrs_g1_u2]), np.mean([mrs_g2_u1,mrs_g2_u2]), np.mean([mrs_g3_u1,mrs_g3_u2])]),4))
print("MRR std (ChatGPT):",np.round(np.std([np.mean([mrs_g1_u1,mrs_g1_u2]), np.mean([mrs_g2_u1,mrs_g2_u2]), np.mean([mrs_g3_u1,mrs_g3_u2])]),4))
#%%
# III. Compute Precision@1, Precision@2, Precision@3 et Precision@4
#%%
# Precision@1, Precision@2, Precision@3 et Precision@4 for AMR:
precision_at_1_u1 = sum([i==1 for i in user_1_1])/len(user_1)
precision_at_1_u2 = sum([i==1 for i in user_2_1])/len(user_1)
print("Precision@1 (AMR):",np.round(np.mean([precision_at_1_u1,precision_at_1_u2]),4))

precision_at_2_u1 = sum([(i==1) or (i==2) for i in user_1_1])/len(user_1)
precision_at_2_u2 = sum([(i==1) or (i==2) for i in user_2_1])/len(user_1)
print("Precision@2 (AMR):",np.round(np.mean([precision_at_2_u1,precision_at_2_u2]),4))

precision_at_3_u1 = sum([(i==1) or (i==2) or (i==3) for i in user_1_1])/len(user_1)
precision_at_3_u2 = sum([(i==1) or (i==2) or (i==3) for i in user_2_1])/len(user_1)
print("Precision@3 (AMR):",np.round(np.mean([precision_at_3_u1,precision_at_3_u2]),4))

precision_at_4_u1 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_1_1])/len(user_1)
precision_at_4_u2 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_2_1])/len(user_1)
print("Precision@4 (AMR):",np.round(np.mean([precision_at_4_u1,precision_at_4_u2]),4))
#%%
# Precision@1, Precision@2, Precision@3 et Precision@4 for ChatGPT:
# GPT-1:
precision_at_1_g1_u1 = sum([i==1 for i in user_1_2])/len(user_1)
precision_at_1_g1_u2 = sum([i==1 for i in user_2_2])/len(user_1)
print("Precision@1 (GPT-1):",np.round(np.mean([precision_at_1_g1_u1,precision_at_1_g1_u2]),4))

precision_at_2_g1_u1 = sum([(i==1) or (i==2) for i in user_1_2])/len(user_1)
precision_at_2_g1_u2 = sum([(i==1) or (i==2) for i in user_2_2])/len(user_1)
print("Precision@2 (GPT-1):",np.round(np.mean([precision_at_2_g1_u1,precision_at_2_g1_u2]),4))

precision_at_3_g1_u1 = sum([(i==1) or (i==2) or (i==3) for i in user_1_2])/len(user_1)
precision_at_3_g1_u2 = sum([(i==1) or (i==2) or (i==3) for i in user_2_2])/len(user_1)
print("Precision@3 (GPT-1):",np.round(np.mean([precision_at_3_g1_u1,precision_at_3_g1_u2]),4))

precision_at_4_g1_u1 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_1_2])/len(user_1)
precision_at_4_g1_u2 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_2_2])/len(user_1)
print("Precision@4 (GPT-1):",np.round(np.mean([precision_at_4_g1_u1,precision_at_4_g1_u2]),4),"\n")

# GPT-2:
precision_at_1_g2_u1 = sum([i==1 for i in user_1_3])/len(user_1)
precision_at_1_g2_u2 = sum([i==1 for i in user_2_3])/len(user_1)
print("Precision@1 (GPT-2):",np.round(np.mean([precision_at_1_g2_u1,precision_at_1_g2_u2]),4))

precision_at_2_g2_u1 = sum([(i==1) or (i==2) for i in user_1_3])/len(user_1)
precision_at_2_g2_u2 = sum([(i==1) or (i==2) for i in user_2_3])/len(user_1)
print("Precision@2 (GPT-2):",np.round(np.mean([precision_at_2_g2_u1,precision_at_2_g2_u2]),4))

precision_at_3_g2_u1 = sum([(i==1) or (i==2) or (i==3) for i in user_1_3])/len(user_1)
precision_at_3_g2_u2 = sum([(i==1) or (i==2) or (i==3) for i in user_2_3])/len(user_1)
print("Precision@3 (GPT-2):",np.round(np.mean([precision_at_3_g2_u1,precision_at_3_g2_u2]),4))

precision_at_4_g2_u1 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_1_3])/len(user_1)
precision_at_4_g2_u2 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_2_3])/len(user_1)
print("Precision@4 (GPT-2):",np.round(np.mean([precision_at_4_g2_u1,precision_at_4_g2_u2]),4),"\n")

# GPT-3:
precision_at_1_g3_u1 = sum([i==1 for i in user_1_4])/len(user_1)
precision_at_1_g3_u2 = sum([i==1 for i in user_2_4])/len(user_1)
print("Precision@1 (GPT-3):",np.round(np.mean([precision_at_1_g3_u1,precision_at_1_g3_u2]),4))

precision_at_2_g3_u1 = sum([(i==1) or (i==2) for i in user_1_4])/len(user_1)
precision_at_2_g3_u2 = sum([(i==1) or (i==2) for i in user_2_4])/len(user_1)
print("Precision@2 (GPT-3):",np.round(np.mean([precision_at_2_g3_u1,precision_at_2_g3_u2]),4))

precision_at_3_g3_u1 = sum([(i==1) or (i==2) or (i==3) for i in user_1_4])/len(user_1)
precision_at_3_g3_u2 = sum([(i==1) or (i==2) or (i==3) for i in user_2_4])/len(user_1)
print("Precision@3 (GPT-3):",np.round(np.mean([precision_at_3_g3_u1,precision_at_3_g3_u2]),4))

precision_at_4_g3_u1 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_1_4])/len(user_1)
precision_at_4_g3_u2 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_2_4])/len(user_1)
print("Precision@4 (GPT-3):",np.round(np.mean([precision_at_4_g3_u1,precision_at_4_g3_u2]),4),"\n")

# Avg (ChatGPT):
precision_at_1_gpt = np.mean([np.mean([precision_at_1_g1_u1,precision_at_1_g1_u2]), np.mean([precision_at_1_g2_u1,precision_at_1_g2_u2]), np.mean([precision_at_1_g3_u1,precision_at_1_g3_u2])])
prec_at_1_gpt_std = np.std([np.mean([precision_at_1_g1_u1,precision_at_1_g1_u2]), np.mean([precision_at_1_g2_u1,precision_at_1_g2_u2]), np.mean([precision_at_1_g3_u1,precision_at_1_g3_u2])])
print("Precision@1 (ChatGPT):",np.round(precision_at_1_gpt,4),"std:",np.round(prec_at_1_gpt_std,4))

precision_at_2_gpt = np.mean([np.mean([precision_at_2_g1_u1,precision_at_2_g1_u2]), np.mean([precision_at_2_g2_u1,precision_at_2_g2_u2]), np.mean([precision_at_2_g3_u1,precision_at_2_g3_u2])])
prec_at_2_gpt_std = np.std([np.mean([precision_at_2_g1_u1,precision_at_2_g1_u2]), np.mean([precision_at_2_g2_u1,precision_at_2_g2_u2]), np.mean([precision_at_2_g3_u1,precision_at_2_g3_u2])])
print("Precision@2 (ChatGPT):",np.round(precision_at_2_gpt,4),"std:",np.round(prec_at_2_gpt_std,4))

precision_at_3_gpt = np.mean([np.mean([precision_at_3_g1_u1,precision_at_3_g1_u2]), np.mean([precision_at_3_g2_u1,precision_at_3_g2_u2]), np.mean([precision_at_3_g3_u1,precision_at_3_g3_u2])])
prec_at_3_gpt_std = np.std([np.mean([precision_at_3_g1_u1,precision_at_3_g1_u2]), np.mean([precision_at_3_g2_u1,precision_at_3_g2_u2]), np.mean([precision_at_3_g3_u1,precision_at_3_g3_u2])])
print("Precision@3 (ChatGPT):",np.round(precision_at_3_gpt,4),"std:",np.round(prec_at_3_gpt_std,4))

precision_at_4_gpt = np.mean([np.mean([precision_at_4_g1_u1,precision_at_4_g1_u2]), np.mean([precision_at_4_g2_u1,precision_at_4_g2_u2]), np.mean([precision_at_4_g3_u1,precision_at_4_g3_u2])])
prec_at_4_gpt_std = np.std([np.mean([precision_at_4_g1_u1,precision_at_4_g1_u2]), np.mean([precision_at_4_g2_u1,precision_at_4_g2_u2]), np.mean([precision_at_4_g3_u1,precision_at_4_g3_u2])])
print("Precision@4 (ChatGPT):",np.round(precision_at_4_gpt,4),"std:",np.round(prec_at_4_gpt_std,4))
#%%
# IV (bonus). Compute Recall@1, Recall@2, Recall@3 et Recall@4 
#%%
# Recall@1, Recall@2, Recall@3 et Recall@4 for AMR:
recall_at_1_u1 = sum([i==1 for i in user_1_1])/len([i for i in user_1_1 if i!=0])
recall_at_1_u2 = sum([i==1 for i in user_2_1])/len([i for i in user_2_1 if i!=0])
print("Recall@1 (AMR):",np.round(np.mean([recall_at_1_u1,recall_at_1_u2]),4))

recall_at_2_u1 = sum([(i==1) or (i==2) for i in user_1_1])/len([i for i in user_1_1 if i!=0])
recall_at_2_u2 = sum([(i==1) or (i==2) for i in user_2_1])/len([i for i in user_2_1 if i!=0])
print("Recall@2 (AMR):",np.round(np.mean([recall_at_2_u1,recall_at_2_u2]),4))

recall_at_3_u1 = sum([(i==1) or (i==2) or (i==3) for i in user_1_1])/len([i for i in user_1_1 if i!=0])
recall_at_3_u2 = sum([(i==1) or (i==2) or (i==3) for i in user_2_1])/len([i for i in user_2_1 if i!=0])
print("Recall@3 (AMR):",np.round(np.mean([recall_at_3_u1,recall_at_3_u2]),4))

recall_at_4_u1 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_1_1])/len([i for i in user_1_1 if i!=0])
recall_at_4_u2 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_2_1])/len([i for i in user_2_1 if i!=0])
print("Recall@4 (AMR):",np.round(np.mean([recall_at_4_u1,recall_at_4_u2]),4))

print("\nLine to copy to .xls:")
print((str(np.round(np.mean([recall_at_1_u1,recall_at_1_u2]),4))+"\t"+str(np.round(np.mean([recall_at_2_u1,recall_at_2_u2]),4))+"\t"+str(np.round(np.mean([recall_at_3_u1,recall_at_3_u2]),4))+"\t"+str(np.round(np.mean([recall_at_4_u1,recall_at_4_u2]),4))).replace('.',','))
#%%
# Recall@1, Recall@2, Recall@3 et Recall@4 for ChatGPT:
# GPT-1:
recall_at_1_g1_u1 = sum([i==1 for i in user_1_2])/len([i for i in user_1_2 if i!=0])
recall_at_1_g1_u2 = sum([i==1 for i in user_2_2])/len([i for i in user_2_2 if i!=0])

recall_at_2_g1_u1 = sum([(i==1) or (i==2) for i in user_1_2])/len([i for i in user_1_2 if i!=0])
recall_at_2_g1_u2 = sum([(i==1) or (i==2) for i in user_2_2])/len([i for i in user_2_2 if i!=0])

recall_at_3_g1_u1 = sum([(i==1) or (i==2) or (i==3) for i in user_1_2])/len([i for i in user_1_2 if i!=0])
recall_at_3_g1_u2 = sum([(i==1) or (i==2) or (i==3) for i in user_2_2])/len([i for i in user_2_2 if i!=0])

recall_at_4_g1_u1 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_1_2])/len([i for i in user_1_2 if i!=0])
recall_at_4_g1_u2 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_2_2])/len([i for i in user_2_2 if i!=0])

# GPT-2:
recall_at_1_g2_u1 = sum([i==1 for i in user_1_3])/len([i for i in user_1_3 if i!=0])
recall_at_1_g2_u2 = sum([i==1 for i in user_2_3])/len([i for i in user_2_3 if i!=0])

recall_at_2_g2_u1 = sum([(i==1) or (i==2) for i in user_1_3])/len([i for i in user_1_3 if i!=0])
recall_at_2_g2_u2 = sum([(i==1) or (i==2) for i in user_2_3])/len([i for i in user_2_3 if i!=0])

recall_at_3_g2_u1 = sum([(i==1) or (i==2) or (i==3) for i in user_1_3])/len([i for i in user_1_3 if i!=0])
recall_at_3_g2_u2 = sum([(i==1) or (i==2) or (i==3) for i in user_2_3])/len([i for i in user_2_3 if i!=0])

recall_at_4_g2_u1 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_1_3])/len([i for i in user_1_3 if i!=0])
recall_at_4_g2_u2 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_2_3])/len([i for i in user_2_3 if i!=0])

# GPT-3:
recall_at_1_g3_u1 = sum([i==1 for i in user_1_4])/len([i for i in user_1_4 if i!=0])
recall_at_1_g3_u2 = sum([i==1 for i in user_2_4])/len([i for i in user_2_4 if i!=0])

recall_at_2_g3_u1 = sum([(i==1) or (i==2) for i in user_1_4])/len([i for i in user_1_4 if i!=0])
recall_at_2_g3_u2 = sum([(i==1) or (i==2) for i in user_2_4])/len([i for i in user_2_4 if i!=0])

recall_at_3_g3_u1 = sum([(i==1) or (i==2) or (i==3) for i in user_1_4])/len([i for i in user_1_4 if i!=0])
recall_at_3_g3_u2 = sum([(i==1) or (i==2) or (i==3) for i in user_2_4])/len([i for i in user_2_4 if i!=0])

recall_at_4_g3_u1 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_1_4])/len([i for i in user_1_4 if i!=0])
recall_at_4_g3_u2 = sum([(i==1) or (i==2) or (i==3) or (i==4) for i in user_2_4])/len([i for i in user_2_4 if i!=0])

# Avg (ChatGPT):
recall_at_1_gpt = np.mean([np.mean([recall_at_1_g1_u1,recall_at_1_g1_u2]), np.mean([recall_at_1_g2_u1,recall_at_1_g2_u2]), np.mean([recall_at_1_g3_u1,recall_at_1_g3_u2])])
recall_at_2_gpt = np.mean([np.mean([recall_at_2_g1_u1,recall_at_2_g1_u2]), np.mean([recall_at_2_g2_u1,recall_at_2_g2_u2]), np.mean([recall_at_2_g3_u1,recall_at_2_g3_u2])])
recall_at_3_gpt = np.mean([np.mean([recall_at_3_g1_u1,recall_at_3_g1_u2]), np.mean([recall_at_3_g2_u1,recall_at_3_g2_u2]), np.mean([recall_at_3_g3_u1,recall_at_3_g3_u2])])
recall_at_4_gpt = np.mean([np.mean([recall_at_4_g1_u1,recall_at_4_g1_u2]), np.mean([recall_at_4_g2_u1,recall_at_4_g2_u2]), np.mean([recall_at_4_g3_u1,recall_at_4_g3_u2])])

print("\nLines to copy to .xls:")
print((str(np.round(np.mean([recall_at_1_g1_u1,recall_at_1_g1_u2]),4))+"\t"+str(np.round(np.mean([recall_at_2_g1_u1,recall_at_2_g1_u2]),4))+"\t"+str(np.round(np.mean([recall_at_3_g1_u1,recall_at_3_g1_u2]),4))+"\t"+str(np.round(np.mean([recall_at_4_g1_u1,recall_at_4_g1_u2]),4))).replace('.',','))
print((str(np.round(np.mean([recall_at_1_g2_u1,recall_at_1_g2_u2]),4))+"\t"+str(np.round(np.mean([recall_at_2_g2_u1,recall_at_2_g2_u2]),4))+"\t"+str(np.round(np.mean([recall_at_3_g2_u1,recall_at_3_g2_u2]),4))+"\t"+str(np.round(np.mean([recall_at_4_g2_u1,recall_at_4_g2_u2]),4))).replace('.',','))
print((str(np.round(np.mean([recall_at_1_g3_u1,recall_at_1_g3_u2]),4))+"\t"+str(np.round(np.mean([recall_at_2_g3_u1,recall_at_2_g3_u2]),4))+"\t"+str(np.round(np.mean([recall_at_3_g3_u1,recall_at_3_g3_u2]),4))+"\t"+str(np.round(np.mean([recall_at_4_g3_u1,recall_at_4_g3_u2]),4))).replace('.',','))
#%%
print("\nLine to copy to .xls:")
print((str(np.round(recall_at_1_gpt,4))+"\t"+str(np.round(recall_at_2_gpt,4))+"\t"+str(np.round(recall_at_3_gpt,4))+"\t"+str(np.round(recall_at_4_gpt,4))).replace('.',','))
#%%
