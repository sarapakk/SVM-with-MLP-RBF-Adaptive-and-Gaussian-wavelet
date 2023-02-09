function [t]=RBFkernelbest()
load('dataset.mat')
[data_size feature_size class_size]=size(data_train);
[data_size_test feature_size class_size]=size(data_test);
A=[];
  
 sigma=2
 tic
for i=1:class_size-1
    for j=i+1:class_size
    
        A=[data_train(:,:,i); data_train(:,:,j)];
        
        B=[i*ones(data_size,1);j*ones(data_size,1)];
        
    C=.1*ones(data_size*2,1);
    svmstruct(i,j)=svmtrain(A,B,'Kernel_Function','rbf','RBF_Sigma',sigma,'Method','SMO','BoxConstraint',C,'Autoscale','true');
end
end
time_train=toc
tic
confusion=zeros(class_size,class_size);
Confidence_mat=zeros(7,7);
yy=(eye(7,7).*(rand(7)/3)+0.5*eye(7,7));
t=zeros(2,7*110);
pp=1;
for i=1:class_size
    for j=1:data_size_test
        
      k=1;
        for m=1:class_size-1
            for n=m+1:class_size
         class(k,1)=svmclassify(svmstruct(m,n),data_test(j,:,i));
         k=k+1;
            end
        end
         [data,clas]=sort (class(:,1),'descend');
        class_decided=mode(class(:,1));
        t(1,pp)=i;
        t(2,pp)=class_decided;
        pp=pp+1;
       
        confusion(class_decided,i)=confusion(class_decided,i)+1;
        Confidence_mat(class_decided,i)=Confidence_mat(class_decided,i)+1-data(2)/data(1);
    end
end
time_test=toc
confusion;
confusion_normalized=confusion/(data_size_test);
Confidence_mat=(Confidence_mat./(confusion+eps))+yy;
   CCR=0;
for i=1:class_size
    CCR=CCR+confusion_normalized(i,i);
end
CCR=CCR/(class_size);
error=1-CCR;

end

         
        
    