functions {
    real genpoiss_lpmf(int y, real theta, real lambda) {
     return log((theta * pow(theta + lambda*y, y-1) * exp(-theta-lambda * y))/tgamma(y+1));
    }
}

data {
  int<lower=1> N;  // total number of observations
  int place[N];  
  int unit_length[N];
}

parameters {
  real<lower = 0> theta1;
  real<lower = -1, upper = 1> lambda1;
  real<lower = 0> mu; 
  real<lower = 0> phi;
  real<lower = 0, upper = 1> psi;  // Probability of observing 0
}

transformed parameters {
  real lprior = 0;
  lprior += gamma_lpdf(theta1 | 2, 0.5); // Specify your prior distribution for theta
  lprior += gamma_lpdf(lambda1 | 1, 1);//  Specify your prior distribution for lambda
  lprior += gamma_lpdf(mu | 1, 1);
  lprior += gamma_lpdf(phi | 1, 1);
}

model {
  for (i in 1:N) {
    if (place[i] == 0) {
      target += log(psi);
    } 
    else {
      target += log(1 - psi);
    }
    target += neg_binomial_2_lpmf(unit_length[i] | mu, phi);
    target += genpoiss_lpmf(place[i] | theta1, lambda1); // Modeling unit_length variable
  }
}

