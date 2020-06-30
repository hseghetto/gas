% ################ Carregar arquivos ################ %
arq_rate_train = load('path_arq_rate_train');
arq_rate_pred = load('path_arq_rate_pred');
arq_pressure_pred = load('path_arq_pressure_pred');
arq_pressure_train = load('path_arq_pressure_train');

% ################## Processing ################# %
ypred = pred(arq_pressure_train, arq_rate_train, arq_rate_pred);
pressure_pred = arq_pressure_train;
disp(pressure_pred)
pressure_pred(:,1) = arq_pressure_train(:,1);
pressure_pred(:,2) = ypred;

derivada_pred = calcular_derivada(pressure_pred);
derivada_test = calcular_derivada(arq_pressure_pred);


% ################ Post-Processing ############## %
loglog(derivada_pred(:,1), derivada_pred(:,2),'o');
hold on 
loglog(derivada_test(:,1), derivada_test(:,2),'*');
xlabel('time (h)')
ylabel('logtime derivative (kgf/cm^{2})')
legend('resultado','dado real')
hold off

% ################################################ %
function ypred = pred(ytrain, xtrain, xpred)

    mxtrain = calcular_xmatrix(xtrain);
    mytrain = calcular_xmatrix(ytrain);
    mxpred  = calcular_xmatrix(xpred);      
    matriz_kernel = calcular_matriz_kernel(mxtrain,mytrain);
    j_beta = calcular_custo_beta(matriz_kernel,xtrain,ytrain(:,2))
    k = calcular_matriz_kernel(mxpred,mxtrain)
    ypred =  k'*j_beta 
    
end


function j_beta = calcular_custo_beta(matriz_kernel,x,y)

    lambda = 2
    beta = calcular_beta(matriz_kernel,y,lambda) 
    delta = 0.5
    j_beta = delta * ( ((( matriz_kernel * beta ) - y ).^2).^-0.5 ) + ((lambda * beta).^2).^-0.5

end


function matriz_kernel = calcular_matriz_kernel(x1,x2)

    c = 1;
    grau_polinomio = 2;
    delta = 1
    matriz_kernel = delta * ( ((x1 * x2')  + c).^grau_polinomio);

end


function beta = calcular_beta(matriz_kernel,y,lambda)

    i = eye(length(matriz_kernel))
    beta = inv(matriz_kernel + (lambda*i)) * y;
    
end


function X = calcular_xmatrix(rate)

    [xrow,xcol] = size(rate); 
    X = zeros(xrow,5);
 
    for i  = 2:xrow

       X(i,1) = 1;  
       for j = 2:i

           X(i,2)= (rate(j,2)-rate(j-1,2)) + X(i,2);
           X(i,3)=((rate(j,2)-rate(j-1,2)) * log10(rate(i,1)-rate(j-1,1))) + X(i,3);
           X(i,4)=((rate(j,2)-rate(j-1,2)) * (rate(i,1)-rate(j-1,1))) + X(i,4);
           X(i,5)=((rate(j,2)-rate(j-1,2)) / (rate(i,1)-rate(j-1,1))) + X(i,5);

           %X(i,6)=((rate(j,2)-rate(j-1,2)) / ( rate(i,1) - rate(j-1,1) )^0.1) + X(i,6);
           %X(i,7)=((rate(j,2)-rate(j-1,2)) * ( rate(i,1) - rate(j-1,1) )^0.1) + X(i,7);
           
       end  
       
    end
    
end


function derivada = calcular_derivada(pressure)

    derivada = pressure; 
    [xrow,xcol] = size(derivada);

    for i = 1:(xrow-1) 
        
        derivada(i,2) =-( (pressure(i+1,2)-pressure(i,2)) ) / ( log(pressure(i+1,1)) - log(pressure(i,1)) );
        
    end

end



