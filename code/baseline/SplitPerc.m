function [X_sel, Y_sel, X_res, Y_res, SelIdx] = SplitPerc(X, Y, percent,seed)

    rng(seed);

    [sample_size, ~] = size(Y);

    SelIdx = randperm(sample_size) <sample_size * percent;
    
    X_sel = X(SelIdx,:);
    Y_sel = Y(SelIdx,:);
    X_res = X(~SelIdx,:);
    Y_res = Y(~SelIdx,:);
    
end