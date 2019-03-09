function [fitness,acc,nfeat]=svm(X,Y,pop)

FeatIndex = find(pop==1);
ntotal_feat=size(X,2);
nfeat=numel(FeatIndex);

X = X(:,[FeatIndex]);

if size(Y,1)<1000
    k=2;
else
    k=5;
end

c = cvpartition(Y,'KFold',k);

acc=[];
pstart=1;
pend=0;

for i =1:k
    pend=pend+c.TestSize(i);
    testset=X([pstart:pend],:);
    trainset=X;
    trainset([pstart:pend],:)=[];
    testlabel=Y(pstart:pend);
    trainlabel=Y;
    trainlabel(pstart:pend)=[];
    model=svmtrain(trainset,trainlabel,'kernel_function','rbf','kktviolationlevel',0.1);
    prdct_label= svmclassify(model,testset);
    acc =[acc, sum(testlabel == prdct_label) / numel(testlabel)];
    pstart=pstart+c.TestSize(i);
end

acc=mean(acc);
fitness=(0.2*(nfeat/ntotal_feat))-0.8*acc;
% fitness=-acc;
