functions {
  real poiss_truncated_lpmf(int y, real lambda, int truncation) {
    
    if (y > truncation) {
      return -1000; // Probability is zero outside the truncation
    }
    return poisson_lpmf(y | lambda);
  }
}

data {
  int<lower=1> N;  // total number of observations
  int place[N];  // response variable
  int unit_length[N];
}

parameters {
  real<lower = 1> lambda;
  real<lower = 0> mu; 
  real<lower = 0> phi; // 
}

transformed parameters {
  real lprior = 0;  // prior contributions to the log posterior
  lprior += gamma_lpdf(lambda | 3, 3);
  lprior += gamma_lpdf(mu | 1, 1);
  lprior += gamma_lpdf(phi | 1, 1);
}

model {
  target += lprior; // prior
  
  for (i in 1:N) {
    target += neg_binomial_2_lpmf(unit_length[i] | mu, phi); 
    target += poiss_truncated_lpmf(place[i] | lambda, unit_length[i]);
  }
}

generated quantities {
  real log_lik[N];
  for (i in 1:N){
    log_lik[i] = neg_binomial_2_lpmf(unit_length[i] | mu, phi);
    log_lik[i] += poiss_truncated_lpmf(place[i] | lambda, unit_length[i]);
  }
}
