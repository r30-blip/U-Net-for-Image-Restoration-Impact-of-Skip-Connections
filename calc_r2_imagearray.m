function r2 = calc_r2_imagearray(true_value, pred_value)

se = squeeze(mean((true_value - pred_value).^2, 1:3)) ;

pred_base = mean(true_value) ;

se_base = squeeze(mean((true_value - pred_base).^2, 1:3)) ;

r2 = 1 - sum(se)/sum(se_base) ;

end