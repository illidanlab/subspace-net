function [loss, W]=single_task_l2(XTRN,YTRN,XTST,YTST,lamb,flag)
    
    if flag
        
        XTRN = [ones(size(XTRN, 1), 1), XTRN];
        XTST = [ones(size(XTST, 1), 1), XTST]; % add for intercept
    end
    
    d = size(XTRN,2);
    LHS = XTRN'*XTRN + lamb*eye(d);
    RHS = XTRN'*YTRN;
    W = LHS\RHS;
    
    YHAT = XTST * W;
    %YHAT(YHAT<0)=0;

    loss = (sum((YHAT-YTST).^2) ./ sum(YTST.^2))';

end
