%svm with adaptive wavelet kernel
clc
clear all
close all
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
k=1;
jj=1;
a1=10000;
cc=1.8*ones(2*data_size,2*data_size);
bequal=.5*ones(1,1);

%svm train 
tic
for c=1:class_size
    for c1=c+1:class_size
a=[data_train1(:,:,c) data_train1(:,:,c1)];
y=[ones(data_size,1); -ones(data_size,1)];

for i=1:2*data_size
    for j=1:2*data_size
        d(i,j)=y(i,:)*y(j,:)*(1-(a(:,i)-a(:,j))'*(a(:,i)-a(:,j))/a1)*exp((a(:,i)-a(:,j))'*(a(:,i)-a(:,j)))*cc(i,j);
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
    if alfa(l,:)>0
         support_vector(:,t,jj)=a(:,l);
         gg=1;
         dd(:,t,jj)=0;
         for l=1:2*data_size
             if ((a(:,l)-support_vector(:,t,jj))'*(a(:,l)-support_vector(:,t,jj)))<10
                 dd(:,t,jj)=(((a(:,l)-support_vector(:,t,jj))'*(a(:,l)-support_vector(:,t,jj))))+dd(:,t,jj);
                 gg=gg+1;
             end
             dd(:,t,jj)=dd(:,t,jj)/gg;
         end
         
         %calculate the c(x)
         cc(i,j)=exp(-(a(:,l)-support_vector(:,t,jj))'*(a(:,l)-support_vector(:,t,jj))/dd(:,t,jj))+cc(i,j); 
         t=t+1;
    end
end
for l=1:2*data_size
    if alfa(l,:)>0
        b(:,c,c1)=(1/y(l,:))-w(:,c,c1)'*a(:,l);
    end
end
k=k+1;
 jj=jj+1;

    end
end
timetrain=toc
ll=1;
% data_test evaluation
tic
Confidence=zeros(7,7);
confusion=zeros(7,7);
for i=1:7
    for j=1:110
        A=7*ones(21,1);
        q=1;
        x=[data_test1(:,j,i)];       
        for m=1:7
            for n=m+1:7
                    g=sign(w(:,m,n)'*x+b(:,m,n));
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
        confusion(i,I)=confusion(i,I)+1;
        Confidence(i,I)=Confidence(i,I)+1-data(2)/data(1);
    end;
end;
ccr=trace(confusion)/(data_size_test*class_size)
confusion
Confusion=confusion/(110)
Confidence_mat=Confidence./(confusion)
timetest=toc
%% ghesmate 3
%1D histogram
figure(3)

hist(data_train(:,11,1))
h = findobj(gca,'Type','patch');
set(h(1,1),'FaceColor','r','EdgeColor','w')
hold on;

hist(data_train(:,11,2))
h = findobj(gca,'Type','patch');
set(h(1,1),'FaceColor','b','EdgeColor','w')
hold on;

hist(data_train(:,11,3))
h = findobj(gca,'Type','patch');
set(h(1,1),'FaceColor','y','EdgeColor','w')
hold on;

hist(data_train(:,11,4))
h = findobj(gca,'Type','patch');
set(h(1,1),'FaceColor','g','EdgeColor','w')
hold on;

hist(data_train(:,11,5))
h = findobj(gca,'Type','patch');
set(h(1,1),'FaceColor','m','EdgeColor','w')
hold on;

hist(data_train(:,11,6))
h = findobj(gca,'Type','patch');
set(h(1,1),'FaceColor','c','EdgeColor','w')
hold on;

hist(data_train(:,11,7))
h = findobj(gca,'Type','patch');
set(h(1,1),'FaceColor','k','EdgeColor','w')
hold on;

legend('W1','W2','W3','W4','W5','W6','W7')
title('histogram of first faeuter')
hold off;

%2D
figure(4)
scatter(data_train(:,11,1),data_train(:,2,1),5,'r');
hold on

scatter(data_train(:,11,2),data_train(:,2,2),5,'b');
hold on

scatter(data_train(:,11,3),data_train(:,2,3),5,'y');
hold on

scatter(data_train(:,11,4),data_train(:,2,4),5,'g');
hold on

scatter(data_train(:,11,5),data_train(:,2,5),5,'m');
hold on

scatter(data_train(:,11,6),data_train(:,2,6),5,'c');
hold on

scatter(data_train(:,11,7),data_train(:,2,7),5,'k');
hold on

legend('W1','W2','W3','W4','W5','W6','W7')
title('2D scatter plot of first two features')
hold off
%3D
figure(5)
scatter3(data_train(:,11,1),data_train(:,2,1),data_train(:,13,1),5,'r');
hold on

scatter3(data_train(:,11,2),data_train(:,2,2),data_train(:,13,2),5,'b');
hold on

scatter3(data_train(:,11,3),data_train(:,2,3),data_train(:,13,3),5,'y');
hold on

scatter3(data_train(:,11,4),data_train(:,2,4),data_train(:,13,4),5,'g');
hold on

scatter3(data_train(:,11,5),data_train(:,2,5),data_train(:,13,5),5,'m');
hold on

scatter3(data_train(:,11,6),data_train(:,2,6),data_train(:,13,6),5,'c');
hold on

scatter3(data_train(:,11,7),data_train(:,2,7),data_train(:,13,7),5,'k');
hold on

legend('W1','W2','W3','W4','W5','W6','W7')
title('3D scatter plot of first three features')
hold off