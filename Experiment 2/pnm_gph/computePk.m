function Pk = computePk(WtW,GRAD,INDk,invINDk,r,m)
% Compute Pk: a subfunction of pnm_nlssubprob

Pk = zeros(r,m); diagWtW = diag(WtW);
if all(invINDk(:))
    Pk = WtW\GRAD;
else
    [sortedinvINDk,sortIx] = sortrows(invINDk');
    breaks = any(diff(sortedinvINDk)');
    breakIx = [0 find(breaks) m];

    for k=1:length(breakIx)-1
        cols = sortIx(breakIx(k)+1:breakIx(k+1));
        invars = invINDk(:,sortIx(breakIx(k)+1));
        vars = INDk(:,sortIx(breakIx(k)+1));
        Pk(invars,cols) = WtW(invars,invars)\GRAD(invars,cols);
        Pk(vars,cols) = GRAD(vars,cols)./repmat(diagWtW(vars),1,length(cols));
    end
end

end