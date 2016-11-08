traindata  = None
i=1
for i,chunk in enumerate(pd.read_csv(dir + 'train_ver2.csv',chunksize=50000,low_memory=False)):
    print(i)
    if traindata is None:
        traindata = chunk.copy()
    else:
        traindata = pd.concat([traindata, chunk])
    del chunk
    gc.collect()
print traindata.info()