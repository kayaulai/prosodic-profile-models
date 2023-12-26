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
  real<lower = 1> lambda1;
  real<lower = 1> lambda2;
}

transformed parameters {
  real lprior = 0;  // prior contributions to the log posterior
  lprior += gamma_lpdf(lambda1 | 10, 10);
  lprior += gamma_lpdf(lambda2 | 5, 5);
}

model {
  target += lprior; // prior
  
  for (i in 1:N) {
    target += poisson_lpmf(unit_length[i] | lambda1);
    target += poiss_truncated_lpmf(place[i] | lambda2, unit_length[i]);
  }
}

generated quantities {
  real log_lik[N];
  for (i in 1:N){
    log_lik[i] = poisson_lpmf(unit_length[i] | lambda1);
    log_lik[i] += poiss_truncated_lpmf(place[i] | lambda2, unit_length[i]);
  }
}

