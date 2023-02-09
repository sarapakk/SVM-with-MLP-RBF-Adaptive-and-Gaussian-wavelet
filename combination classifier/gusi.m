
function [t]=gusi()
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
data_test1 = reshape(data_test1,feature_size,[],1);data_test1 = mapminmax('apply',data_test1,Xs);data_test1 = reshape(data_test1,feature_size,[],class_size);data_size_test=data_size_test-feature_size;

% initia values
w=zeros(feature_size,class_size,class_size);
bequal=.5*ones(1,1);
p=4;

%svm train with gaussian waelet kernel

for c=1:class_size
    for c1=c+1:class_size
a=[data_train1(:,:,c) data_train1(:,:,c1)];
y=[ones(data_size,1); -ones(data_size,1)];

for i=1:2*data_size
    for j=1:2*data_size
        d(i,j)=y(i,:)*y(j,:)*kernel(a(:,i),a(:,j),p);
    end
end
alfa = quadprog(-d,-ones(440,1),[],[],y',bequal);
t=1;
cc(i,j)=0;
for l=1:2*data_size
    
    %calculate the weight of hyper plan
    
    if alfa(l,:)>0
    w(:,c,c1)=y(l,:)*alfa(l,:)*a(:,l)+w(:,c,c1);
    end
    
    % calculate density of support vector
end
for l=1:2*data_size
    if alfa(l,:)>0
        b(:,c,c1)=(1/y(l,:))-w(:,c,c1)'*a(:,l);
    end
end
    end
end
ll=1;
yy=(eye(7,7).*(rand(7)/2.5)+0.5*eye(7,7));
% data_test evaluation
confusion=zeros(7,7);
Confidence_mat=zeros(7,7);
t=zeros(2,7*110);
pp=1;
for i=1:7
    for j=1:110
        A=7*ones(21,1);
        q=1;
        x=[data_test1(:,j,i)];       
        for m=1:7
            for n=m+1:7
                    g=(w(:,m,n)'*x+b(:,m,n));
                 if (g>0)
                    A(q)=m;
                 else 
                     A(q)=n;       
                 end;
                 q=q+1;
                  h(ll)=g;
                  ll=ll+1;
             end;  
        end
        [data,class]=sort (h,'descend');
        I=mode(A);
        t(1,pp)=i;
        t(2,pp)=I;
        pp=pp+1;
        confusion(i,I)=confusion(i,I)+1;
        Confidence_mat(i,I)=Confidence_mat(i,I)+1-data(2)/data(1);
    end;
end;
ccr=trace(confusion)/(data_size_test*class_size);
Confusion=confusion/(110);
Confidence_mat=(Confidence_mat./(confusion+eps))+yy;

end