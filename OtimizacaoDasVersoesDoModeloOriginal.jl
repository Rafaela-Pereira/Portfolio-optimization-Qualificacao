using CSV;
using DataFrames;
using JuMP;
using Ipopt;
using Plots;
#----------------------------------------
#ÍNDICES

#Ativos - i ∈ {1,..,M}. M=12
#Dias   - t ∈ {1,...,N}.N=313
#perfis - j ∈ {1,...,L}. L=3
M=12
N=503
L=3

#--------------------------------------------------
#PARÂMETROS

#Taxa de retorno diário dos ativos. Matriz N x M
tabelaRetorno=CSV.File("C://Users//Rafaela//Desktop//Documentos Gerais//Tese-Finanças//dados//retornoFinal1.csv");
DataRetornos = DataFrame(tabelaRetorno);
retornos=zeros(N,M)
retornos.=DataRetornos[:,2:13]


#Perfil de risco. Vetor de tamanho L
χ=[1/3 1/3 1/3]

#nível de confiança.Vetor de tamanho L
α=[0.6 , 0.75,  0.9 ]

#retorno mínimo aceitável pelo investidor
δ=0.3

#Domínio do x(pesos)
inf_x=0.1
sup_x=0.8

#-----------------------------------------------------------------
#    Teste 0 - O modelo original 
#-----------------------------------------------------------------


modelT0 = Model(Ipopt.Optimizer)

    @variables(modelT0, begin
    1 ≥ x[1:M] ≥ 0
        u[1:N] ≥ 0
        y[1:L] 
        z[1:N,1:L] ≥ 0
        end)
   
 
    @constraint(modelT0, [i=1:M],sup_x ≥ x[i] ≥ inf_x);
    @constraint(modelT0, u[1]==0 );
    @constraint(modelT0,[t=2:N], u[t] ≥ u[t-1] - transpose(x)*retornos[t,:]);
    @constraint(modelT0, sum(transpose(x)*retornos[t,:] for t=1:N) ≥ δ );
    @constraint(modelT0,[t=1:N,j=1:L], u[t]-y[j] ≤ z[t,j]);
  
    @objective(modelT0, Min,  sum(χ[j]*(y[j]+ 1/((1-α[j])*N)*sum(z[t,j] for t=1:N )) for j=1:L));
   
    t0=time()
    optimize!(modelT0) #otimizar
    Δt0=time()-t0
 #solução
    pesoT0= value.(x); pesoT0
    VaRT0= value.(y); VaRT0
    zT0= value.(z);
   uT0=value.(u);
   CVaRmistoT0=round(objective_value(modelT0);digits=3)


#-------------------------------------------------------------------------------
    #MODELO 1 - VERSÃO 1-  função Max aproximada pela raiz(f_1), fornecendo as derivadas 
#--------------------------------------------------------------------------------

modelT1 = Model(Ipopt.Optimizer)

    @variables(modelT1, begin
    1 ≥ x[1:M] ≥ 0
        u[1:N] ≥ 0
        y[1:L] 
        end)
    
    @constraint(modelT1, [j=1:M],sup_x ≥ x[j] ≥ inf_x);
    @constraint(modelT1, u[1]==0 );
    @constraint(modelT1,[t=2:N], u[t] ≥ u[t-1] - transpose(x)*retornos[t,:]);
    @constraint(modelT1, sum(transpose(x)*retornos[t,:] for t=1:N) ≥ δ );
   
    # função substituta f1--------------------------------
    η=2^(-14)        #parâmetro da função f1, menor melhor
   
   
    f1(a)=  (a +sqrt(a^2+η))/2      #a função  f1
  
    df1(a)= 1/2*(a/(sqrt(a^2+η))+1)       # a derivada da função f1
    
    d2f1(a)= η/(2*(η+a^2)^(3/2))    # a derivada segunda da função f1
  
    #-------------------------------------------------------
    
    JuMP.register(modelT1, :f1, 1, f1, df1,  d2f1)
   
    @NLobjective(modelT1, Min,  sum(χ[j]*(y[j]+ 1/((1-α[j])*N)*sum(f1(u[t]-y[j]) for t=1:N )) for j=1:L)); 
    
    t1=time()
    optimize!(modelT1) #otimizar
    Δt1=time()-t1;
    #-------
    #solução
    pesoT1= value.(x); pesoT1
    VaRT1= value.(y); VaRT1
    uT1=value.(u);
    CVaRmistoT1=round(objective_value(modelT1);digits=3)
#-------------------------------------------------------------------------------
    #MODELO 2 - VERSÃO 2-função aproximada da Max por partes(f_3), fornecendo as derivadas
#--------------------------------------------------------------------------------

modelT2 = Model(Ipopt.Optimizer)

    @variables(modelT2, begin
    1 ≥ x[1:M] ≥ 0
        u[1:N] ≥ 0
        y[1:L] 
        end)
    
    @constraint(modelT2, [i=1:M],sup_x ≥ x[i] ≥ inf_x);
    @constraint(modelT2, u[1]==0 );
    @constraint(modelT2,[t=2:N], u[t] ≥ u[t-1] - transpose(x)*retornos[t,:]);
    @constraint(modelT2, sum(transpose(x)*retornos[t,:] for t=1:N) ≥ δ );
   
     # função substituta f2--------------------------------
    η = 0.03  #parâmetro da função f2, menor melhor
    f2(a)=   # a função f2
    if a ≤ - η
        result= 0.0
    elseif - η < a ≤ η
        result = (a^2)/(4*η) + a/2 + η/4
    elseif  a > η
        result = a
    end

    df2(a)=   #a derivada da função f2
    if a ≤ - η
        result= 0.0
    elseif - η < a ≤ η
        result = a/(2*η) + 1/2 
    elseif  a > η
        result = 1.0
    end

    d2f2(a)=   # a segunda derivada da função f2
    if a ≤ - η
        result= 0.0
    elseif - η < a ≤ η
        result = 1/(2*η) 
    elseif  a > η
        result = 0.0
    end
    #----------------

JuMP.register(modelT2, :f2, 1, f2, df2, d2f2)
 
@NLobjective(modelT2, Min,  sum(χ[j]*(y[j]+ 1/((1-α[j])*N)*sum(f2(u[t]-y[j]) for t=1:N )) for j=1:L)); 

t2=time()
optimize!(modelT2) #otimizar
Δt2=time()-t2

#------------
#solução

pesoT2= value.(x); pesoT2
VaRT2= value.(y); VaRT2
uT2=value.(u);
CVaRmistoT2=round(objective_value(modelT2);digits=3)
#------------------------------------------------------------------------------
#  MODELO 3 -VERSÃO 3- função aproximada da Max por partes(f_3), fornecendo as derivadas
#----------------------------------------------------------------------------

modelT3 = Model(Ipopt.Optimizer)

    @variables(modelT3, begin
    1 ≥ x[1:M] ≥ 0
        u[1:N] ≥ 0
        y[1:L] 
        end)
    
    @constraint(modelT3, [i=1:M],sup_x ≥ x[i] ≥ inf_x);
    @constraint(modelT3, u[1]==0 );
    @constraint(modelT3,[t=2:N], u[t] ≥ u[t-1] - transpose(x)*retornos[t,:]);
    @constraint(modelT3, sum(transpose(x)*retornos[t,:] for t=1:N) ≥ δ );
   
     # função substituta f3--------------------------------
    η = 0.021  #parâmetro da função f3, menor melhor
    f3(a)=    #a função f3
    if a ≤ - η
        result= 0.0 
    elseif - η < a ≤ η
        result = (-a^4)/(16*η^3)+ (3*a^2)/(8*η)+ a/2+ (3*η)/16 
    elseif  a > η
        result = a 
    end
    
    df3(a)=   #a derivada da função f3
    if a ≤ - η
        result= 0.0
    elseif - η < a ≤ η
        result = (-a^3)/(4*η^3)+ (3*a)/(4*η)+ 1/2
    elseif  a > η
        result = 1.0
    end

    d2f3(a)=   # a segunda derivada da função f3
    if a ≤ - η
        result= 0.0
    elseif - η < a ≤ η
        result = (-3*a^2)/(4*η^3)+ 3/(4*η)
    elseif  a > η
        result = 0.0
    end
    #------------

JuMP.register(modelT3, :f3, 1, f3, df3, d2f3)
 
@NLobjective(modelT3, Min,  sum(χ[j]*(y[j]+ 1/((1-α[j])*N)*sum(f3(u[t]-y[j]) for t=1:N )) for j=1:L)); 

t3=time()
optimize!(modelT3) #otimizar
Δt3=time()-t3

#----------------
#solução
pesoT3= value.(x); pesoT3
VaRT3= value.(y);  VaRT3
uT3 =value.(u); 
CVaRmistoT3=round(objective_value(modelT3);digits=3)

#-------------------------------------------------------------------
    #MODELO 4 -VERSÃO 4- função aproximada da Max pela NORMA, fornecendo as derivadas
#---------------------------------------------------------------------

modelT4 = Model(Ipopt.Optimizer)

    @variables(modelT4, begin
    1 ≥ x[1:M] ≥ 0
        u[1:N] ≥ 0
        y[1:L] 
        end)
    
    @constraint(modelT4, [j=1:M], sup_x ≥ x[j] ≥ inf_x);
    @constraint(modelT4, u[1]==0 );
    @constraint(modelT4,[t=2:N], u[t] ≥ u[t-1] - transpose(x)*retornos[t,:]);
    @constraint(modelT4, sum(transpose(x)*retornos[t,:] for t=1:N) ≥ δ );
   
    # função substituta f4--------------------------------
    β=4
    η = 0.0000027
    f4(a)= (a+ ((a)^β+η)^(1/β))/2    #a função f4
    
 
    df4(a)=1/2*(a^(β-1)*(a^β+ η)^(1/β-1)+1)

    d2f4(a)=1/2*((β-1)*η*a^(β-2)*(a^β+η)^(1/β-2))

   
    #--------------------------------------

JuMP.register(modelT4, :f4, 1, f4, df4, d2f4)
 
@NLobjective(modelT4, Min,  sum(χ[j]*(y[j]+ 1/((1-α[j])*N)*sum(f4(u[t]-y[j]) for t=1:N )) for j=1:L)); 

t4=time()
optimize!(modelT4) #otimizar
Δt4=time()-t4
#solução
pesoT4= value.(x); pesoT4  #pesos
VaRT4= value.(y);  VaRT4  #var
uT4 =value.(u);  #drawdowns
CVaRmistoT4=round(objective_value(modelT4);digits=3)
#------------------------------------------------------------------------
    #MODELO 5 - VERSÃO 5- função aproximada da Max por partes, fornecendo derivadas
#-------------------------------------------------------------------------

modelT5 = Model(Ipopt.Optimizer)

    @variables(modelT5, begin
    1 ≥ x[1:M] ≥ 0
        u[1:N] ≥ 0
        y[1:L] 
        end)
    
    @constraint(modelT5, [i=1:M],sup_x ≥ x[i] ≥ inf_x);
    @constraint(modelT5, u[1]==0 );
    @constraint(modelT5,[t=2:N], u[t] ≥ u[t-1] - transpose(x)*retornos[t,:]);
    @constraint(modelT5, sum(transpose(x)*retornos[t,:] for t=1:N) ≥ δ );
   
     # função substituta f5--------------------------------
      η = 0.01 #parâmetro da f5
      f5(a)=    #a função f5
      if a ≤ - η
          result= 0.0
      elseif - η < a ≤ η
          result = 1/2*(a*(-1/(2*η^3)*a^3+3*a/(2*η))+a)   
      elseif  a > η
          result = a 
      end
 
      df5(a)=    #a derivada da função f5
      if a ≤ - η
          result= 0.0
      elseif - η < a ≤ η
          result = -a^3/η^3+3*a/(2*η) +1/2
      elseif  a > η
          result = 1.0
      end

      d2f5(a)=    #a derivada segunda da função f5
      if a ≤ - η
          result= 0.0
      elseif - η < a ≤ η
          result = 3*(η^2 -2*a^2)/(2*η^3)
      elseif  a > η
          result = 0.0
      end
     #----------------------------------------

JuMP.register(modelT5, :f5, 1, f5, df5, d2f5)
 
@NLobjective(modelT5, Min,  sum(χ[j]*(y[j]+ 1/((1-α[j])*N)*sum(f5(u[t]-y[j]) for t=1:N )) for j=1:L)); 

t5=time()
optimize!(modelT5) #otimizar
Δt5=time()-t5;
#solução
pesoT5= value.(x);   #pesos
VaRT5= value.(y); VaRT5   #var
uT5 =value.(u);  #drawdowns
CVaRmistoT5=round(objective_value(modelT5);digits=3)
#round(pesoT7[12];digits=3) 
#-------------------------------------------------------------------
    #MODELO 6 - VERSÃO 6-função aproximada da Max  x^p/(x^(p-1)+a),fornecendo as derivadas
#---------------------------------------------------------------------

modelT6 = Model(Ipopt.Optimizer)

    @variables(modelT6, begin
    1 ≥ x[1:M] ≥ 0
        u[1:N] ≥ 0
        y[1:L] 
        end)
    
    @constraint(modelT6, [i=1:M],sup_x ≥ x[i] ≥ inf_x);
    @constraint(modelT6, u[1]==0 );
    @constraint(modelT6,[t=2:N], u[t] ≥ u[t-1] - transpose(x)*retornos[t,:]);
    @constraint(modelT6, sum(transpose(x)*retornos[t,:] for t=1:N) ≥ δ );

   # função substituta f6--------------------------------
      ρ = 3 #parâmetro
      η= 1e-5
      f6(a)=    #a função f6
      if a ≥ 0.0
          result= a^(ρ)/(a^(ρ-1)+η)
      elseif  a < 0.0
          result = 0.0 
       end
     
       df6(a)=    #a  derivada  da função f6
      if a ≥ 0.0
          result= (a^ρ*(η*ρ*a+a^ρ))/(η*a+a^ρ)^2
      elseif  a < 0.0
          result = 0.0 
       end
    
      d2f6(a)=    #a derivada segunda da função f6
      if a ≥ 0.0
        result= η*(1-ρ)*a^ρ*((ρ-2)*a^ρ-η*ρ*a)/(η*a+a^ρ)^3
    elseif  a < 0.0
        result = 0.0 
     end
    #------------------------------

JuMP.register(modelT6, :f6, 1, f6,  df6,d2f6)
 
@NLobjective(modelT6, Min,  sum(χ[j]*(y[j]+ 1/((1-α[j])*N)*sum(f6(u[t]-y[j]) for t=1:N )) for j=1:L)); 

t6=time()
optimize!(modelT6) #otimizar
Δt6=time()-t6;
#solução
pesoT6= value.(x); pesoT6;  #pesos
VaRT6= value.(y); VaRT6   #var
uT6 =value.(u);  #drawdowns
CVaRmistoT6=round(objective_value(modelT6);digits=3)
#-----------------------------------------------------------------
#Resultados finais 

#Tempos
Tempos=zeros(7)
Tempos[1]=round(Δt0;digits=3) 
Tempos[2]=round(Δt1;digits=3) 
Tempos[3]=round(Δt2;digits=3) 
Tempos[4]=round(Δt3;digits=3) 
Tempos[5]=round(Δt4;digits=3) 
Tempos[6]=round(Δt5;digits=3) 
Tempos[7]=round(Δt6;digits=3) 
Tempos

#pesos (x)
pesos=zeros(12,7)

pT0=zeros(length(pesoT0))
for i=1:length(pesoT0)
pT0[i]=round(pesoT0[i]; digits=3)
end

pT1=zeros(length(pesoT1))
for i=1:length(pesoT1)
pT1[i]=round(pesoT1[i]; digits=3)
end

pT2=zeros(length(pesoT2))
for i=1:length(pesoT2)
pT2[i]=round(pesoT2[i]; digits=3)
end

pT3=zeros(length(pesoT3))
for i=1:length(pesoT3)
pT3[i]=round(pesoT3[i]; digits=3)
end

pT4=zeros(length(pesoT4))
for i=1:length(pesoT4)
pT4[i]=round(pesoT4[i]; digits=3)
end

pT5=zeros(length(pesoT5))
for i=1:length(pesoT5)
pT5[i]=round(pesoT5[i]; digits=3)
end

pT6=zeros(length(pesoT6))
for i=1:length(pesoT6)
pT6[i]=round(pesoT6[i]; digits=3)
end

pesos[:,1]= pT0; pesos[:,2]= pT1;pesos[:,3]= pT2;
pesos[:,4]= pT3;pesos[:,5]= pT4;pesos[:,6]= pT5;
pesos[:,7]= pT6;
pesos[7,:]

#Var's

VaRs=zeros(3,7)

vT0=zeros(length(VaRT0))
for i=1:length(VaRT0)
vT0[i]=round(VaRT0[i]; digits=3)
end

vT1=zeros(length(VaRT1))
for i=1:length(VaRT1)
vT1[i]=round(VaRT1[i]; digits=3)
end

vT2=zeros(length(VaRT2))
for i=1:length(VaRT2)
vT2[i]=round(VaRT2[i]; digits=3)
end

vT3=zeros(length(VaRT3))
for i=1:length(VaRT3)
vT3[i]=round(VaRT3[i]; digits=3)
end

vT4=zeros(length(VaRT4))
for i=1:length(VaRT4)
vT4[i]=round(VaRT4[i]; digits=3)
end

vT5=zeros(length(VaRT5))
for i=1:length(VaRT5)
vT5[i]=round(VaRT5[i]; digits=3)
end

vT6=zeros(length(VaRT6))
for i=1:length(VaRT6)
vT6[i]=round(VaRT6[i]; digits=3)
end

VaRs[:,1]= vT0;VaRs[:,2]= vT1;VaRs[:,3]= vT2;
VaRs[:,4]= vT3;VaRs[:,5]= vT4;VaRs[:,6]= vT5;
VaRs[:,7]= vT6;
VaRs[3,:]

#CVaR's misto
CVaRs=zeros(7)
CVaRs[1]=CVaRmistoT0 ;
CVaRs[2]=CVaRmistoT1;
CVaRs[3]=CVaRmistoT2; 
CVaRs[4]=CVaRmistoT3; 
CVaRs[5]=CVaRmistoT4;
CVaRs[6]=CVaRmistoT5; 
CVaRs[7]=CVaRmistoT6; 
CVaRs


#--------------------------------------------------------
#               Modelo escolhido: VERSÃO 1- fronteira eficiente
#-------------------------------------------------------

function otimizarCDDsujeitoAoRetorno(γ)

modelT1 = Model(Ipopt.Optimizer)

    @variables(modelT1, begin
    1 ≥ x[1:M] ≥ 0
        u[1:N] ≥ 0
        y[1:L] 
        end)
    
    @constraint(modelT1, [j=1:M],sup_x ≥ x[j] ≥ inf_x);
    @constraint(modelT1, u[1]==0 );
    @constraint(modelT1,[t=2:N], u[t] ≥ u[t-1] - transpose(x)*retornos[t,:]);
    @constraint(modelT1, sum(transpose(x)*retornos[t,:] for t=1:N) ≥ γ );
   
    # função substituta f2--------------------------------
    η=2^(-14)        #parâmetro da função f2, menor melhor

    f2(a)=  (a +sqrt(a^2+η))/2      #a função  f2
  
    df2(a)= 1/2*(a/(sqrt(a^2+η))+1)       # a derivada da função f2

    d2f2(a)= η/(2*(η+a^2)^(3/2))    # a derivada segunda da função f2
  
    #-------------------------------------------------------
    
    JuMP.register(modelT1, :f2, 1, f2, df2,  d2f2)
   
    @NLobjective(modelT1, Min,  sum(χ[j]*(y[j]+ 1/((1-α[j])*N)*sum(f2(u[t]-y[j]) for t=1:N )) for j=1:L)); 
    

    optimize!(modelT1) #otimizar
      #-------

    #solução
    pesoT1= value.(x); pesoT1
    VaRT1= value.(y); VaRT1
    CVaRmistoT1=round(objective_value(modelT1);digits=3)
    return pesoT1, VaRT1, CVaRmistoT1
end 

#0.65, 1.12    !!!!
otimizarCDDsujeitoAoRetorno(1.6) #0.053
#fronteira eficiente - Conjunto de pontos (γ, retorno acumulado)

#γ1=[0.3;0.5; 0.55; 0.62;  0.7;0.8;0.85;0.95;1.05;1.13;1.18;1.36]
γ1=[0.3;0.4; 0.62;0.8;0.95;1.05;1.13;1.18;1.36;1.5;1.6;1.7;1.8;2.0;2.2]

CDD1=zeros(length(γ1))
for i=1:length(γ1)
    CDD1[i]=otimizarCDDsujeitoAoRetorno(γ1[i])[3]
end
CDD1




#gráficos - fronteira eficiente
plot(CDD1, γ1,  leg=false)
scatter!(CDD1, γ1,  leg=false)


#gráfico -risco retorno ajustado
riscoRetAjustado1=(CDD1.^(-1)).*γ1
plot(γ1, riscoRetAjustado1,leg=false)
scatter!(γ1, riscoRetAjustado1, leg=false)

#---------------------------------------------------
#MODELO ORIGINAL-fronteira eficiente

function otimizarCDDsujeitoAoRetornoOriginal(γ)

modelT0 = Model(Ipopt.Optimizer)

    @variables(modelT0, begin
    1 ≥ x[1:M] ≥ 0
        u[1:N] ≥ 0
        y[1:L] 
        z[1:N,1:L] ≥ 0
        end)
   
    #@variable(modelT0, A[1:N,1:L] ≥ 0)
 
    @constraint(modelT0, [i=1:M],sup_x ≥ x[i] ≥ inf_x);
    @constraint(modelT0, u[1]==0 );
    @constraint(modelT0,[t=2:N], u[t] ≥ u[t-1] - transpose(x)*retornos[t,:]);
    @constraint(modelT0, sum(transpose(x)*retornos[t,:] for t=1:N) ≥ γ );
    @constraint(modelT0,[t=1:N,j=1:L], u[t]-y[j] ≤ z[t,j]);
  
    @objective(modelT0, Min,  sum(χ[j]*(y[j]+ 1/((1-α[j])*N)*sum(z[t,j] for t=1:N )) for j=1:L));
   
    t0=time()
    optimize!(modelT0) #otimizar
    Δt0=time()-t0
 #solução
    pesoT0= value.(x); pesoT0
    VaRT0= value.(y); VaRT0
    zT0= value.(z);
   uT0=value.(u);
   CVaRmistoT0=round(objective_value(modelT0);digits=3)

   return pesoT0, VaRT0, CVaRmistoT0
end 

γ0=[0.3;0.4; 0.62;0.8;0.95;1.05;1.13;1.18;1.36;1.5;1.6;1.7;1.8;2.0;2.2]

CDD0=zeros(length(γ0))
for i=1:length(γ0)
    CDD0[i]=otimizarCDDsujeitoAoRetornoOriginal(γ0[i])[3]
end
CDD0