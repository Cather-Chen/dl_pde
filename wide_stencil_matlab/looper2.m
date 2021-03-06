function [u,resRec,errMat,time,count] = looper2(F,g,n,N,levels,iterVec,h,u0,xa,xb,ya,yb,tol,mex)
tic
count = 0;
u = u0;
res = 1;
while res > tol
    count = count + 1;
    [u,resMat,err] = FAS_V2(F,g,n,N,levels,iterVec,h,u,xa,xb,ya,yb,count,mex);
    res = norm(resMat(:),inf); 
    if count == 1
        errMat = err;
        resRec = resMat;
    else
        errMat = cat(3,errMat,err);
        resRec = cat(3,resRec,resMat);
    end
    
end

time = toc;

end