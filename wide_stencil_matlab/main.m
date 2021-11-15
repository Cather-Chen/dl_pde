close all
clear all
f1 = @(x,y) (1 + x.^2).*(1 + y.^2).*exp(x.^2 + y.^2);
f2 = @(x,y) exp(x.^2 + y.^2).*(1+.5*(x+y).^2).*(1+.5*(y-x).^2);
g = @(x,y) exp((x.^2 + y.^2)/2);

minN = 4;
maxN = 7;
nValNum = length(minN:maxN);

stats = zeros(nValNum,maxN,2);
nVec = 2.^(minN+1:maxN+1)+1;
hVec = 2.^(-(minN+1:maxN+1));

errCell = cell(nValNum,maxN);
resCell = cell(nValNum,maxN);
frameCell = cell(nValNum,maxN);


exactSolCell = cell(nValNum,1);
basesNum = 2;
iterVec = [5 50];

for i = minN:maxN
    n = 2^(i+1) + 1;
    h = 2/(n-1);
    xa = -1; xb = 1; ya = -1; yb = 1; tol = h^2/10;
    [X,Y] = meshgrid(xa:h:xb,ya:h:yb);
    if basesNum == 1
        F = f1(X,Y);
    else
        F1 = f1(X,Y);
        F2 = f2(X,Y);
        F = min(F1,F2);
    end

    G = g(X,Y);
    exactSolCell{i-minN+1} = G;
    A = zeros(n);
    u0 = init(F,g,n,h,X,Y);

    u0(:,1) = g(X(:,1),Y(:,1));
    u0(:,n) = g(X(:,n),Y(:,n));
    u0(1,:) = g(X(1,:),Y(1,:));
    u0(n,:) = g(X(n,:),Y(n,:)); 

    N = n;

    for j = 1:4
        [u,resMat,errMat,time,count] = looper2(F,g,n,N,j,2*iterVec,h,u0,xa,xb,ya,yb,tol,0);
        stats(i-minN+1,j,1) = norm(errMat(:,:,end),inf);
        stats(i-minN+1,j,2) = count;
        errFrames = unique(ceil(linspace(1,count,4)));
        frameCell{i-minN+1,j} = errFrames;
        errCell{i-minN+1,j} = errMat(:,:,errFrames);
        resCell{i-minN+1,j} = resMat(:,:,errFrames);

    end

end
   
%% Error and iteration number plots

legendStrs = {'One level','Two levels','Three levels','Four levels',...
    'Five levels','Six levels','Seven levels','Eight levels',...
    'Nine levels','Ten levels'};
errFig = figure;
plot(hVec,stats(:,1,1), 'o-')
xlabel('h')
ylabel('Error')
title('Error vs. h for all depths of recursion')
axis tight
saveas(errFig,'errFig.fig')

countFig = figure;
semilogy(hVec,stats(:,:,2),'o-');
legend(legendStrs(1:4));
xlabel('h')
ylabel('Iterations')
title(sprintf('Number of iterations vs. h for %d depths of recursion',4))
axis tight
saveas(countFig,'countFig.fig')

%% Error surface plots

errorDir = 'error_surfs';
mkdir(errorDir);
resDir = 'res_surfs';
mkdir(resDir);
exactSolDir = 'exact_sol_surfs';
mkdir(exactSolDir);

for i = 1:nValNum
    
    nValDir1 = sprintf('%s/N_%d',errorDir,1/hVec(i)+1);
    mkdir(nValDir1)
    
    nValDir2 = sprintf('%s/N_%d',resDir,1/hVec(i)+1);
    mkdir(nValDir2)

    for j = 1:maxN
        if ~isempty(errCell{i,j}) 
            depthDir1 = sprintf('%s/depth_%d',nValDir1,j);
            mkdir(depthDir1)

            depthDir2 = sprintf('%s/depth_%d',nValDir2,j);
            mkdir(depthDir2)

            subplotNum = size(errCell{i,j},3);
            err = errCell{i,j};
            res = resCell{i,j};
            
            for k = 1:subplotNum

                fig1 = figure;
                surf(linspace(-1,1,nVec(i)),linspace(-1,1,nVec(i)),abs(err(:,:,k)),...
                    'linestyle','none');
                title(sprintf('h = %f, depth = %d levels, iteration = %d',...
                    hVec(i),j,frameCell{i,j}(k)));
%                 zlim([0 norm(err(:),inf)]);
                saveas(fig1,sprintf('%s/count_%d.fig',depthDir1,frameCell{i,j}(k)));

                fig2 = figure;
                surf(linspace(-1,1,nVec(i)),linspace(-1,1,nVec(i)),res(:,:,k),...
                    'linestyle','none');
                title(sprintf('h = %f, depth = %d levels, iteration = %d',...
                    hVec(i),j,frameCell{i,j}(k)));
%                 zlim([0 norm(res(:),inf)]);
                saveas(fig2,sprintf('%s/count_%d.fig',depthDir2,frameCell{i,j}(k))); 

            end
        end
    end
    
    fig = figure;
    surf(exactSolCell{i},'linestyle','none');
    title(sprintf('Exact solution evaluated for h = %f',hVec(i)));
    saveas(fig,sprintf('%s/N_%d.fig',exactSolDir,1/hVec(i)+1));
end