% Particle Swarm Optimisation algorithm for Feature Selection by Sadegh Salesi and Georgina Cosma      %
% Programmed by Sadegh Salesi at Nottignham Trent University              %
% Last revised:  2017     %
% Reference: S. Salesi and G. Cosma, A novel extended binary cuckoo search algorithm for feature selection, 2017 2nd International Conference on Knowledge Engineering and Applications (ICKEA), London, 2017, pp. 6-12.
% https://ieeexplore.ieee.org/document/8169893
% Copyright (c) 2017, Sadegh Salesi and Georgina Cosma. All rights reserved.
% -----------------------------------------------------------------

clc
clear
close all
format shortG

%% parameters setting 
for nrun=1:10
X=xlsread('data_heart');
Y=xlsread('target_heart');

nvar=size(X,2);
lb=0*ones(1,nvar); % lower bound
ub=1*ones(1,nvar);  % upper bound

fen=10000;
popsize=30; % population size
maxiter=floor(fen/popsize); % max of iteation

w=1;
c1=1;
c2=2;
wdamp=0.9;

%% initial population algorithm
tic
emp.var=[];
emp.acc=[];
emp.fit=[];
emp.vel=[];
emp.selected=[];
emp.nfeat=[];
emp.t=[];

par=repmat(emp,popsize,1);
for i=1:popsize
  par(i).var=lb+rand(1,nvar).*(ub-lb);
  [par(i).fit,par(i).acc,par(i).nfeat]=svm(X,Y,round(par(i).var));
  par(i).vel=0;
end  
    

bpar=par;
[value,index]=min([par.fit]);
gpar=par(index);
    
%% main loop algorithm
BEST=zeros(maxiter,1);
tic
for iter=1:maxiter
     for i=1:popsize
         par(i).vel=w*par(i).vel+...
                    c1*rand(1,nvar).*(bpar(i).var-par(i).var)+...
                    c2*rand(1,nvar).*(gpar.var-par(i).var);
                
        par(i).vel=min(par(i).vel,0.1);
        par(i).vel=max(par(i).vel,-0.1);
        
        par(i).var=par(i).var+par(i).vel;
        

        par(i).var=min(par(i).var,ub);
        par(i).var=max(par(i).var,lb);
        
        if sum(par(i).var)>0
            [par(i).fit,par(i).acc,par(i).nfeat]=svm(X,Y,round(par(i).var));
        else
            par(i).fit=Inf;
        end
        
        
        if par(i).fit<bpar(i).fit
            bpar(i)=par(i);
            
            if bpar(i).fit<gpar.fit
                gpar=bpar(i);
            end
        end

     end



BEST(iter)=gpar.fit;
disp([' Run = ' num2str(nrun) ' Iter = '  num2str(iter)  ' BEST = '  num2str(BEST(iter))  ' Acc = ' num2str(gpar.acc) ' nfeat = ' num2str(gpar.nfeat) ])
w=w*wdamp;
end

save(nrun,1)=gpar.acc;
save(nrun,2)=gpar.nfeat;
save(nrun,3)=toc;

end

%% results algorithm

% disp([ ' Best Solution = ' num2str(gpar.info.x) ])
% disp([ ' Best Fitness = ' num2str(gpar.fit) ])
% disp([ ' Time = ' num2str(toc) ])
% 
% 
% figure(1)
% plot(BEST,'r')
% xlabel('Iteration ')
% ylabel(' Fitness ')
% legend('BEST')
% title('PSO')



