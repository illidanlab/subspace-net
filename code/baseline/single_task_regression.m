
function [loss, W]=single_task_regression(XTRN,YTRN,XTST,YTST,flag)
    
    if flag        
        XTRN = [ones(size(XTRN, 1), 1), XTRN];
        XTST = [ones(size(XTST, 1), 1), XTST]; % add for intercept
    end
    
    LHS = XTRN'*XTRN;
    RHS = XTRN'*YTRN;
    W = LHS\RHS;
    
    YHAT = XTST * W;
    %YHAT(YHAT<0)=0;

    loss = (sum((YHAT-YTST).^2) ./ sum(YTST.^2))';     

end
