clc
clear all
close all
load('dataset.mat')
[data_size feature_size class_size]=size(data_train);
[data_size_test feature_size class_size]=size(data_test);
A=[];
s=1;
CCR=0;   
 p1=.1
 p2=-2
 tic
for i=1:class_size-1
    for j=i+1:class_size
    
        A=[data_train(:,:,i); data_train(:,:,j)];
        
        B=[i*ones(data_size,1);j*ones(data_size,1)];
        
    C=.1*ones(data_size*2,1);
    svmstruct(i,j)=svmtrain(A,B,'Kernel_Function','mlp','MLP_Params',[p1 p2],'Method','SMO','BoxConstraint',C,'Autoscale','true');
end
end
time_train=toc
tic
confusion=zeros(class_size,class_size);
for i=1:7
    for j=1:110
      k=1;
        for m=1:class_size-1
            for n=m+1:class_size
         class(k,1)=svmclassify(svmstruct(m,n),data_test(j,:,i));
         k=k+1;
            end
        end
        class_decided=mode(class(:,1));
        confusion(class_decided,i)=confusion(class_decided,i)+1;
    end
end
time_test=toc
confusion
confusion_normalized=confusion/(data_size_test)
   
for i=1:class_size
    CCR=CCR+confusion(i,i);
end
CCR=CCR/(class_size*data_size_test)
error=1-CCR
         
      
        
    