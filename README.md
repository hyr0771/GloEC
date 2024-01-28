# GloEC: a hierarchical-aware global model for enzyme function prediction

## Abstract
The annotation of enzyme function is a fundamental challenge in industrial biotechnology and pathologies. Numerous computational methods have been proposed to predict enzyme function by annotating enzyme labels with Enzyme Commission number. However, the existing methods face difficulties in modelling the hierarchical structure  of enzyme label in a global view. Moreover, they cannot fully utilize the mutual interactions between different levels  of enzyme label.Here, we formulate the hierarchy of enzyme label as a directed enzyme graph and propose a hierarchical-aware structure encoder to globally model enzyme label depen dency. Based on the enzyme hierarchy encoder, we develop an end-to-end hierarchical-aware global model named GloEC to predict enzyme function.  GloEC learns hierarchical-aware enzyme label embeddings via hierarchy-GCN(Graph Convolutional Network) encoder to capture the interactions of different enzyme label levels and performs deductive fusion of label-aware enzyme features to annotate enzyme function. Thorough comparative experiments on three different datasets show that GloEC achieves better predictive performance as compared to the existing methods. The case studies also demonstrate that GloEC is capable of effectively predicting the function of isoenzyme.

##In the Code：

##In the Datasets：

##about
title = {GloEC: a hierarchical-aware global model for enzyme function prediction}
