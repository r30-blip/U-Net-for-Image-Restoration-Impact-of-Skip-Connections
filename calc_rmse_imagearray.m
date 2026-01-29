function rmse = calc_rmse_imagearray(true_value, pred_value)

se = squeeze(mean((true_value - pred_value).^2, 1:3)) ;

mse = mean(se) ;

rmse = sqrt(mse) ;

end