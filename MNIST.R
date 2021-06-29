rm(list=ls())
library('plot3D')
library('roccv')
teste <- read.csv("trainReduzido.csv")
treino <- read.csv("validacao.csv")

cat("Dimensao dos dados de teste:",dim(teste), "\n")
cat("Dimensao dos dados de treino:",dim(treino), "\n")


teste_id <- teste[,1] #separa o id
#teste_id <- as.factor(teste_id)
#summary(teste_id)
teste_id <- as.matrix(teste_id)

teste_label <- teste[,2] #separa o label
#teste_label <- as.factor(teste_label)
teste_label <- as.matrix(teste_label)

Xteste <- teste[,-c(2,2)]
dados_pixel_teste <- teste[,-c(2,1)] #separa os pixels

treino_id <- treino[,1] #separa o id
#treino_id <- as.factor(treino_id)
#summary(treino_id)

Xtreino <- treino[,-c(2,2)]
dados_pixel_treino <- treino[,-c(1,1)] #separa os pixels

library(caret)


sample_n<-4000
training<-teste_label[1:sample_n, ]

#Matriz de confusão 

set.seed(2019)
digits_mlp<-mlp(as.matrix(training[,-1]),
                decodeClassLabels(training$n),
                size=40,learnFunc="Rprop",
                shufflePatterns = F,
                maxit=60)

mlp_p<-predict(digits_mlp,as.matrix(Test[,-1]))
d_mlp_p<-encodeClassLabels(mlp_p,method="WTA",l=0,h=0)-1
caret::confusionMatrix(xtabs(~d_mlp_p+Test$n))


# Multi Layer Perceptron:
MLPerceptron <- function(xin, yd, eta, tol, maxepocas, neuronios, xtest, ytest) {
  dimxin <- dim(xin)
  N <- dimxin[1]
  n <- dimxin[2]
  
  wo <- matrix( runif( (n+1)*neuronios, -0.5, 0.5), nrow =neuronios, ncol=n+1 )
  wt <- matrix(runif(neuronios+1)-0.5, nrow = 1)
  xin <- cbind(1, xin)
  xtest <- cbind(1, xtest)
  
  nepocas <- 0
  eepoca <- tol + 1
  
  evec <- matrix(0, nrow = 1, ncol = maxepocas)
  eTestvec <- matrix(0, nrow = 1, ncol = maxepocas)
  while((nepocas < maxepocas) && (eepoca > tol)) {
    erro <- errotest <- 0
    xseq <- sample(N)
    
    for(i in 1:N) {
      irand <- xseq[i]
      
      z1 <- wo %*% xin[irand, ]
      a1 <- rbind(1, tanh(z1))
      
      z2 <- wt %*% a1
      #yhati <- tanh(z2)
      yhati <- z2
      
      e <- yd[irand]-yhati
      deltaE2 <- -1*e
      dwt <- eta*deltaE2 %*% t(a1)
      
      dwo <- matrix(0,dim(wo)[1], dim(wo)[2])
      for(i in 1:dim(wo)[1]) {
        dwo[i,] <- ( eta*deltaE2*wt[,i+1]*( 1/cosh(z1[i,])^2 ) ) %*% t(xin[irand, ])
      }
      
      wt <- wt - dwt
      wo <- wo - dwo
      erro <- erro + e*e
    }
    
    xtestseq <- sample(dim(xtest)[1])
    for(i in 1:dim(xtest)[1]) {
      irandtest <- xtestseq[i]
      Z1test <- wo %*% xtest[irandtest, ]
      A1test <- tanh(Z1test)
      Yhattest <- wt %*% rbind(1,A1test)
      Predict <- Yhattest
      etest <- ytest[irandtest] - Predict
      errotest <- errotest + etest*etest
    }
    
    nepocas <- nepocas + 1
    
    evec[nepocas] <- erro/N
    eTestvec[nepocas] <- errotest/dim(xtest)[1]
    
    eepoca <- evec[nepocas]
    
    cat("Erro[", nepocas, "]: ", evec[nepocas], "\n")
  }
  retlist <- list(wo, wt, evec[1:nepocas], eTestvec[1:nepocas])
  return(retlist)
}

MLPredict <- function(xin, model) {
  W1 <- model[[1]]
  W2 <- model[[2]]
  X <- cbind(1,xin)
  Predict <- matrix(0, dim(xin)[1])
  
  for(i in 1:dim(X)[1]) {
    Z <- W1 %*% X[i,]
    A <- tanh(Z)
    Yhat <- W2 %*% rbind(1,A)
    Predict[i] <- Yhat
  }
  return(Predict)
}
