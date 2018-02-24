function [loss]=single_task_l1(XTRN,YTRN,XTST,YTST,lamb,flag)
    
    l = length(lamb);
    [n, t] = size(YTST);  
    loss = zeros(t,l);
    
    %XTST = [ones(size(XTST, 1), 1), XTST]; % add for intercept
    for i=1:t
        [B, fitInfo] = lasso(XTRN,YTRN(:,i),'Lambda',lamb);
        YHAT = XTST * B;
        bias = repmat(fitInfo.Intercept, n ,1);
        YHAT = YHAT + bias;
        %YHAT(YHAT<0)=0;
        
        YTST_l = repmat(YTST(:,i), 1, l);
        loss(i,:) = sum((YHAT-YTST_l).^2) ./ sum(YTST_l.^2);
        
    end


end
