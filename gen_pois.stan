functions {
    real genpoiss_lpmf(int y, real theta, real lambda) {
     return log(theta)+
            (y-1)*log(theta+lambda*y)
            -lgamma(y+1)-theta-lambda*y;
    }
}

data {
  int<lower=1> N;  // total number of observations
  int place[N];  // response variable
  int unit_length[N];
}

parameters {
  real<lower = 1> theta1;
  real<lower = 0> lambda1;
  real<lower = 1> theta2;
  real<lower = 0> lambda2;
}

transformed parameters {
  real lprior = 0;
  lprior += gamma_lpdf(theta1 | 2, 0.5); // Specify your prior distribution for theta
  lprior += gamma_lpdf(lambda1 | 1, 1);//  Specify your prior distribution for lambda
  lprior += gamma_lpdf(theta2 | 2, 0.5); // Specify your prior distribution for theta
  lprior += gamma_lpdf(lambda2 | 1, 1);//  Specify your prior distribution for lambda
}

model {
  for (i in 1:N) {
    place[i] ~ genpoiss_lpmf(theta1, lambda1);  // Using the generalized Poisson distribution
    unit_length[i] ~ genpoiss_lpmf(theta2, lambda2);   // Modeling unit_length variable
  }
  
}
