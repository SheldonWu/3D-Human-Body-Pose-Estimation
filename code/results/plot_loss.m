% plot loss
loss = csvread('log_numLayers5_epochs12000_drop0.6_lr0.001.csv');

subplot(3,1,1)
plot(loss(:,1))
ylim([0 5000])
ylabel('L2 Loss')
title('2D to 3D Training Loss')
set(gca, 'FontSize', 18)
grid on

subplot(3,1,2)
plot(loss(:,2))
ylabel('Mean Training RMSE')
set(gca, 'FontSize', 18)
grid on

subplot(3,1,3)
plot(loss(:,3))
xlabel('Epoch')
ylabel('Mean Testing RMSE')
set(gca, 'FontSize', 18)
grid on


