function [loss, W]=multi_task_l21(XTRN,YTRN,XTST,YTST,lamb,flag)
    
    addpath('../MALSAR/MALSAR/functions/joint_feature_learning/'); % load function
    addpath('../MALSAR/MALSAR/utils/'); % load utilities
    
    %opts.init = 0;      % guess start point from data.
    opts.tFlag = 1;     % terminate after relative objective value does not changes much.
    opts.tol = 10^-5;   % tolerance. 
    opts.maxIter = 1500; % maximum iteration number of optimization.

    LHS = XTRN'*XTRN;
    RHS = XTRN'*YTRN;
    opts.W0 = LHS\RHS;

    [~, t]=size(YTST);

    if flag
        
        XTRN = [ones(size(XTRN, 1), 1), XTRN]; % add for intercept
        XTST = [ones(size(XTST, 1), 1), XTST]; % add for intercept
    end
    
    XTRN_cell = cell(1,t);
    for i=1:t
        XTRN_cell{i} = XTRN;
    end
    
    YTRN_cell = cell(1,t);
    for i=1:t
        YTRN_cell{i} = YTRN(:,i);
    end
    
    [W, ~] = Least_L21(XTRN_cell, YTRN_cell, lamb, opts);
    YHAT = XTST * W;

    %YHAT(YHAT<0)=0;

    loss = (sum((YHAT-YTST).^2) ./ sum(YTST.^2))';
end
