%Part 1
Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;

V = linspace(-1.95,0.7,200);

I1 = Is*(exp((1.2*V)/0.025)-1) + Gp*V - Ib*(exp((-1.2*(V+Vb))/0.025)-1);

variation = linspace(-0.2,0.2,41);

I2 = I1;

for i = 1:length(I2)
    error = variation(randi(length(variation)));
    I2(i) = I2(i)*(1+error);
end

f1 = figure(1);
plot(V,I1,'b','DisplayName','Calculated Diode Current')
legend('Location','northwest')
title('Diode Current without Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f1,'northwest')

f2 = figure(2);
plot(V,I2,'b','DisplayName','Calculated Diode Current')
legend('Location','northwest')
title('Diode Current with Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f2,'northeast')

f5 = figure(5);
semilogy(V,abs(I1),'b','DisplayName','Calculated Diode Current')
legend('Location','northwest')
title('Diode Current without Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f5,'northwest')

f6 = figure(6);
semilogy(V,abs(I2),'b','DisplayName','Calculated Diode Current')
legend('Location','northwest')
title('Diode Current with Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f6,'northeast')

%Part 2
fourthOrder1 = polyfit(V,I1,4);
fourthOrder2 = polyfit(V,I2,4);
eigthOrder1 = polyfit(V,I1,8);
eigthOrder2 = polyfit(V,I2,8);

fourthI1 = polyval(fourthOrder1,V);
fourthI2 = polyval(fourthOrder2,V);
eigthI1 = polyval(eigthOrder1,V);
eigthI2 = polyval(eigthOrder2,V);

figure(1)
hold on
plot(V,fourthI1,'g','DisplayName','Fitted Fourth Order No Variation')
plot(V,eigthI1,'m','DisplayName','Fitted Eigth Order No Variation')
legend('Location','northwest')
movegui(f1,'northwest')
hold off

figure(2)
hold on
plot(V,fourthI2,'g','DisplayName','Fitted Fourth Order with Variation')
plot(V,eigthI2,'m','DisplayName','Fitted Eigth Order with Variation')
legend('Location','northwest')
movegui(f2,'northeast')
hold off

figure(5)
hold on
semilogy(V,abs(fourthI1),'g','DisplayName','Fitted Fourth Order No Variation')
semilogy(V,abs(eigthI1),'m','DisplayName','Fitted Eigth Order No Variation')
legend('Location','northwest')
movegui(f5,'northwest')
hold off

figure(6)
hold on
semilogy(V,abs(fourthI2),'g','DisplayName','Fitted Fourth Order with Variation')
semilogy(V,abs(eigthI2),'m','DisplayName','Fitted Eigth Order with Variation')
legend('Location','northwest')
movegui(f6,'northeast')
hold off

%Part 3
fo = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');

V = transpose(V);
I1 = transpose(I1);
I2 = transpose(I2);

NLfitI1 = fit(V,I1,fo);
NLfitI2 = fit(V,I2,fo);

nlI1 = NLfitI1(V);
nlI2 = NLfitI2(V);
%{
f3 = figure(3);
plot(V,I1,'b','DisplayName','Calculated Diode Current')
hold on
plot(V,nlI1,'k','DisplayName','4-Parameter Non-Linear Fit')
legend('Location','northwest')
title('Diode Current without Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f3,'southwest')

f4 = figure(4);
plot(V,I2,'b','DisplayName','Calculated Diode Current')
hold on
plot(V,nlI2,'k','DisplayName','4-Parameter Non-Linear Fit')
legend('Location','northwest')
title('Diode Current with Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f4,'southeast')
%}

fa = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
NLfitI1A = fit(V,I1,fa);
NLfitI2A = fit(V,I2,fa);

nlI1A = NLfitI1A(V);
nlI2A = NLfitI2A(V);

f3 = figure(3);
plot(V,I1,'b','DisplayName','Calculated Diode Current')
hold on
plot(V,nlI1A,'m','DisplayName','2-Parameter Non-Linear Fit')
legend('Location','northwest')
title('Diode Current without Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f3,'southwest')

f4 = figure(4);
plot(V,I2,'b','DisplayName','Calculated Diode Current')
hold on
plot(V,nlI2A,'m','DisplayName','2-Parameter Non-Linear Fit')
legend('Location','northwest')
title('Diode Current with Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f4,'southeast')

f7 = figure(7);
semilogy(V,abs(I1),'b','DisplayName','Calculated Diode Current')
hold on
semilogy(V,abs(nlI1A),'m','DisplayName','2-Parameter Non-Linear Fit')
legend('Location','northwest')
title('Diode Current without Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f7,'southwest')

f8 = figure(8);
semilogy(V,abs(I2),'b','DisplayName','Calculated Diode Current')
hold on
semilogy(V,abs(nlI2A),'m','DisplayName','2-Parameter Non-Linear Fit')
legend('Location','northwest')
title('Diode Current with Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f8,'southeast')

fb = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
NLfitI1B = fit(V,I1,fb);
NLfitI2B = fit(V,I2,fb);

nlI1B = NLfitI1B(V);
nlI2B = NLfitI2B(V);

figure(3)
hold on
plot(V,nlI1B,'c','DisplayName','3-Parameter Non-Linear Fit')
legend('Location','northwest')
title('Diode Current without Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f3,'southwest')

figure(4)
hold on
plot(V,nlI2B,'c','DisplayName','3-Parameter Non-Linear Fit')
legend('Location','northwest')
title('Diode Current with Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f4,'southeast')

figure(7)
hold on
semilogy(V,abs(nlI1B),'c','DisplayName','3-Parameter Non-Linear Fit')
legend('Location','northwest')
title('Diode Current without Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f7,'southwest')

figure(8)
hold on
semilogy(V,abs(nlI2B),'c','DisplayName','3-Parameter Non-Linear Fit')
legend('Location','northwest')
title('Diode Current with Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f8,'southeast')

%Part 4

net1 = cascadeforwardnet([10,8,6,4,2]);
net2 = net1;
inputs = transpose(V);
outputs1 = transpose(I1);
outputs2 = transpose(I2);

net1 = train(net1,inputs,outputs1);
net2 = train(net2,inputs,outputs2);

netI1 = net1(inputs);
netI2 = net2(inputs);

figure(3)
hold on
plot(inputs,netI1,'r','DisplayName','Neural Net Model')
legend('Location','northwest')
title('Diode Current without Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f3,'southwest')

figure(4)
hold on
plot(inputs,netI2,'r','DisplayName','Neural Net Model')
legend('Location','northwest')
title('Diode Current with Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f4,'southeast')

figure(7)
hold on
semilogy(inputs,abs(netI1),'r','DisplayName','Neural Net Model')
legend('Location','northwest')
title('Diode Current without Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f7,'southwest')

figure(8)
hold on
semilogy(inputs,abs(netI2),'r','DisplayName','Neural Net Model')
legend('Location','northwest')
title('Diode Current with Error')
xlabel('Voltage (V)')
ylabel('Current (A)')
movegui(f8,'southeast')



