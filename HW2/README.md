This data set---which consists of 16087 samples with 10013 features---is a subset of E2006-tfidf [1].

The data is stored in a 'hw2.mat' file, which is a structure as follows.
    "structure with fields:
        X: [16087x10013 double]
        y: [16087x1 double]
    "
    
Notice that we store the data matrix X in a sparse way for faster transmission. You should convert it to full storage before implement your algorithm.



[1] https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#E2006-tfidf