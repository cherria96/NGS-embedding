# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:59:50 2022

@author: Sujin
"""

import pandas as pd 
from tkinter import filedialog
import numpy as np
import os

#CSV file download from 'silva_16S_barplot.qzv' with Taxonomic level 7
parent_dir = "../data/qiimeresult/ARC/"
filename = parent_dir + "level-7.csv"
# metadata = "../fastq/sample-metadata-"+ domain.lower() +".tsv"
table = parent_dir + "metadata.tsv"

#filename = filedialog.askopenfilename(title = "Open taxa file")
#metadata = filedialog.askopenfilename(title = "Open metadata file .tsv")
#table = filedialog.askopenfilename(title = "ASV_table.tsv")

name= 'sampleid'

# namemap = pd.read_csv(metadata, sep = '\t', index_col = 0)
# namemap = namemap.sort_values(by=name)
# namemap = namemap.astype('str')
# #namemap.loc[namemap['reactor'] == 'r2', 'days'] = namemap.loc[namemap['reactor'] == 'r2', 'days'] + '-2'
# namemap_columns = len(namemap.columns)

data = pd.read_csv(filename, index_col = 0 , na_values= ['',' - ']).T
data = data.iloc[:-3]
data.index = data.index.astype('str')
name_split = data.index.str.split(";")
data = data.astype('int')


col_names = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
for col, i in zip(col_names, range(0,7)):
    data.insert(i, col, name_split.str.get(i))
    data[col] = data[col].str.split('__').str[1]
data = data.replace('', np.NaN)
reads = data.iloc[:, 7:]
row_count = reads.iloc[:,-1].count()

unid = ['uncultured', 'unidentified', 'Ambiguous', 'metagenome', 'Unknown', 'group']
for j in range(0, 7):
    for i in range(0, row_count):
        if any(f in str(data.iloc[i,j]) for f in unid):
            data.iloc[i,j] = 'unidentified'
            
data = data.fillna('unidentified')
data.reset_index(inplace = True)
data = data.replace(to_replace = ';', value = '; ', regex = True)

ASV = pd.read_csv(table, sep = '\t')[1:]
OUT = pd.concat([ASV, data],axis = 1, join= 'inner')
OUT.drop(['Confidence', 'index', 'Taxon'], axis = 1, inplace = True)

file = parent_dir + "taxa_result.xlsx"
OUT.to_excel(excel_writer=file, sheet_name = "OTUs", index = False)

OUT = pd.read_excel(file, index_col = 0, na_values= ['',' - ',0])

reads = OUT.iloc[:, 7:]
reads_sum = reads.sum(axis = 0).to_frame()
row_count = reads.iloc[:,-1].count()
col_count = reads.count(1)[1]

Number = range(1,7)
Name = ['P_read', 'C_read', 'O_read', 'F_read', 'G_read', 'S_read']
Name_per = ['P(%)', 'C(%)', 'O(%)', 'F(%)', 'G(%)', 'S(%)']
Name_major=['P_rank(%)', 'C_rank(%)', 'O_rank(%)', 'F_rank(%)', 'G_rank(%)', 'S_rank(%)']
Number_name = list(zip(Number, Name, Name_per, Name_major))


def num_to_per(df1, df2, row_count, col_count, r_sum):
    for i in range(0, row_count):
        for j in range(1, col_count+1):
            per = df1.iloc[i,j]/r_sum.iloc[j-1][0]*100
            df2.iloc[i,j] = round(per,3)
    return (df2)

def drop_minor(df):
    for idx, row in df.iterrows():
        if row.max() < 1:
            df.drop(idx, inplace = True)
    return (df)
        

for i, j, p, r in Number_name:
    new1 = pd.merge(OUT.iloc[:, i], reads, left_index = True, right_index = True, how = 'left')
    new2 = new1.copy()
    new2 = num_to_per(new1, new2, row_count, col_count, reads_sum)
    new1 = new1.groupby(new1.iloc[:,0]).sum()
    new2 = new2.groupby(new2.iloc[:,0]).sum()
    new3 = new2.copy()
    new3 = drop_minor(new3)
    with pd.ExcelWriter(file, mode = 'a', engine = 'openpyxl') as writer:
        new1.to_excel(writer, sheet_name = j)
        new2.to_excel(writer, sheet_name = p)
        new3.to_excel(writer, sheet_name = r)

