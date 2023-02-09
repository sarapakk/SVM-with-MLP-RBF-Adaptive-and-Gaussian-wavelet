close all
clear all
clc
load('dataset.mat')
[data_size feature_size class_size]=size(data_train);
[data_size_test feature_size class_size]=size(data_test);
sum=zeros(feature_size,class_size);
sum_cov=zeros(feature_size,feature_size,class_size);
for i=1:7
    data_train1(:,:,i)=data_train(:,:,i)';
data_test1(:,:,i)=data_test(:,:,i)';
end

% normalize data

FP.ymin = 0; FP.ymax = 1;
data_train1 = reshape(data_train1,feature_size,[],1);[data_train1, Xs]= mapminmax(data_train1,FP);data_train1 = reshape(data_train1,feature_size,[],class_size);
data_test1 = reshape(data_test1,feature_size,[],1);data_test1 = mapminmax('apply',data_test1,Xs);data_test1 = reshape(data_test1,feature_size,[],class_size);data_size_test=data_size_test-2*class_size;

RBF=RBFkernelbest();
MLP=mlpbest();
GUSSIAN=gusi();
WAVE=wavelet();
c=zeros(1,4);
d=zeros(2,110*7);
conf=zeros(7,7);

for i=1:110*7
    c(1,1)=RBF(2,i);
    c(1,2)=MLP(2,i);
    c(1,3)=WAVE(2,i);
    c(1,4)=GUSSIAN(2,i);
    d(2,i)=mode(c);
    d(1,i)=RBF(1,i);
        conf(d(1,i),d(2,i))=conf(d(1,i),d(2,i))+1;
end
conf
confusion_normalized=conf/(110)
ccr=trace(conf)/(data_size_test*class_size)

