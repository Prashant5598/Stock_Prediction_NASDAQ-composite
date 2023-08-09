clc
clear all
close all
%% Reading the file and Distribution analysis
data = xlsread("NASDAQ_DATASET.csv");
histogram(data, 10)
mean(data)
median(data)
std(data)
skewness(data)
kurtosis(data)
[h,p] = jbtest(data)

%% Splitting the data
training_data = data(1:487);
testing_data = data(488:506);

%% Testing for stationarity
[h, p] = adftest(training_data)

%% Transforming the data
n = length(data);
data_new = [];
for i=2:n
data_new=[data_new;100*(data(i,1)-data(i-1,1))/data(i-1,1)];
end

%% Testing for stationarity
[h, p] = adftest(data_new(1:487))

%% Plotting the transformed data and original data 
figure
subplot(1,2,1)
plot(training_data)
subplot(1,2,2)
plot(data_new(1:487))

%% Determining the order of the ARIMA model using Graphical method
figure 
subplot(1,2,1)
autocorr(data_new(1:487))
subplot(1,2,2)
parcorr(data_new(1:487))

%% Determining the order of the ARIMA model using BOX Jenkins method
maxorder=5;
AIC=[];
BIC=[];
for i=0:maxorder
    for j=0:maxorder
        model = arima(i,1,j); %i is the index of the AR order, j is the index of the MA order - in each matrix the element ij corresponds to a ARMA(i-1,j-1) model
        [EstMdl,EstParamCov,logL,info] = estimate(model,training_data,'Display','off');
        [aic,bic] = aicbic(logL,length(info.X),size(training_data,1));
        AIC(i+1,j+1)=aic;
        BIC(i+1,j+1)=bic;
    end
end
AIC
min(min(AIC)) %AIC generally suggests larger models
BIC
min(min(BIC)) 

%% Creating text output for both (note, that if more elements have the same value, just the first one is displayed - pay attention to the matrixes as well!)
[M,I] = min(AIC(:));
[I_row, I_col] = ind2sub(size(AIC),I);
OUTPUT=['AIC: ARMA(' , num2str(I_row-1) , ',' , num2str(I_col-1) , ')'];
disp(OUTPUT)
[M,I] = min(BIC(:));
[I_row, I_col] = ind2sub(size(BIC),I);
OUTPUT=['BIC: ARMA(' , num2str(I_row-1) , ',' , num2str(I_col-1) , ')'];
disp(OUTPUT)

%% Creating the model and checking the model
model1 = arima(4,1,4);
[EstMdl,EstParamCov,logL,info] = estimate(model1, training_data);
[Y, YMSE] = forecast(EstMdl, 19, "Y0", training_data);
output = smoothts(Y, "e", 19)
figure
h1 = plot(488:506, data(488:506))
hold on
h2 = plot(488:506, output,'b','LineWidth',1);
h3 = plot(488:506,Y + 1.96*sqrt(YMSE),'r:',...
		'LineWidth',2);
plot(488:506,Y - 1.96*sqrt(YMSE),'r:','LineWidth',2);
legend([h1 h2 h3],'Observed','Forecast',...
		'95% Confidence Interval','Location','NorthWest');
title(['19-days Forecasts'])
hold off
H(:,1) = Y + 1.96*sqrt(YMSE);
H(:,2)= Y - 1.96*sqrt(YMSE);
MSE = (sum(data(488:506) - Y).^2)/19
MAE  = (sum(abs(data(488:506)) - Y))/19
RMSE = sqrt(mean((data(488:506) - Y).^2))

%% Evaluating the Final model
[E, V] = infer(EstMdl, training_data);
figure 
subplot(1,2,1)
autocorr(E)
subplot(1,2,2)
parcorr(E)
[h,P_val] = lbqtest(E) 
